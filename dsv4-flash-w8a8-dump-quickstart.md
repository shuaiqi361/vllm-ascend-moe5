# DeepSeek-V4 Activation Dump — Quick Start (for agents)

What the data is, how it's structured, and how to load it.

## What this is
Intermediate activations from `deepseek-v4-w8a8-mtp` (a MoE transformer: 4096 hidden dim, 256 routed experts, top-6 routing). Dumped **per sequence, per generated token, per routed-MoE layer**. Each dumped row is one generated token's values at one layer.

## Layout
```
<root>/metadata.json                   # config + what's in the files
<root>/seq_000000/seq_meta.json         # prompt_token_ids, output_token_ids
<root>/seq_000000/token_00000.pt        # one generated token → dict of 160 tensors (prefill)
<root>/seq_000000/token_00001.pt
...
<root>/seq_000001/...
```
- `seq_{i:06d}` = one input sequence; `token_{j:05d}.pt` = activations that produced its j-th output token.
- `token 0` = prefill (the last-prompt-position row, predicting the first output token); `token k` = decode step k. The **number of `token_*.pt` files = number of generated tokens**; enumerate by globbing (see below).
- Layers present: indices **3–42** (40 routed-MoE layers; 0–2 are hash/dense and absent). Layer tag = `L{idx:02d}` → `L03 … L42`.

## The tensors (per layer, per token)
Each is the single token's **1-D row**. Dict key = `"{layer_tag}.{target}"`, e.g. `"L17.router_logits"`. 40 layers × 5 core targets = **200 entries** per file. **Decode** tokens additionally carry up to 40 `lrc_cache_hit` entries (one per layer) → up to 240; **prefill** (`token_00000`) has the 200 core.

| target | shape | dtype | meaning |
|---|---|---|---|
| `router_input` | `(4096,)` | bf16 | residual-stream activation fed into the MoE gate at that layer |
| `pre_attn_input` | `(4096,)` | bf16 | post-LayerNorm activation fed into attention at that layer |
| `router_logits` | `(256,)` | fp32 | **raw** gate scores over the 256 routed experts (pre-scoring, pre-topk) |
| `router_bias` | `(256,)` | fp32 | per-expert selection bias `e_score_correction_bias`, added to `sigmoid(router_logits)` to pick experts (group top-k); weights use the *unbiased* scores. Layer-static (same every token). Present every token. |
| `topk_ids` | `(6,)` | int16 | the 6 routed experts selected (logical ids 0–255; **order not meaningful**; shared experts excluded) |
| `lrc_cache_hit` | `(6,)` | bool | **decode tokens only.** Per-routed-expert offload-cache residency, **aligned 1:1 with `topk_ids`** (here order *is* meaningful): `1` = expert was already in the LRC expert cache when the layer needed it, `0` = miss (paged in on demand). E.g. all six resident → `[1,1,1,1,1,1]`. |

`router_logits` are raw — `topk_ids` is **not** `argtop6(sigmoid(router_logits))`. The real selection adds `router_bias` to the sigmoid scores, then does grouped top-k: `topk_ids ≈ group_topk(sigmoid(router_logits) + router_bias)`. The routing weights use the *unbiased* sigmoid scores (renormalized). So to reconstruct the selection you need `router_logits` **and** `router_bias` (plus `metadata.model.num_expert_group` / `topk_group`).

## Metadata
- `metadata.json` (root): `model` dims (`hidden_size`, `n_routed_experts`, `num_experts_per_tok`, `scoring_func`, `dumped_layer_indices`), `target_order` (the five core target names), `num_expert_group`/`topk_group`/`norm_topk_prob`/`routed_scaling_factor` (routing hyperparams, under `model`), `optional_targets` (`["lrc_cache_hit"]` — decode-only), `targets` (dim + dtype per name, including `lrc_cache_hit`), `layer_tags` (`["L03"…"L42"]`), `key_scheme`, `layout` (what each file/dir is), `file_format`, and `dump_options` (e.g. `max_tokens_per_seq`).
- `seq_meta.json` (per sequence): `prompt_token_ids`, `output_token_ids`. Note `len(output_token_ids)` = (number of token files) **− 1** — the final generated token is never seen as a later input, so it isn't listed. Get the token count by globbing `token_*.pt`, not from this list.

## Load it
```python
import os, json, torch

def load_root(root):        return json.load(open(os.path.join(root, "metadata.json")))
def seq_meta(root, i):      return json.load(open(os.path.join(root, f"seq_{i:06d}", "seq_meta.json")))
def token_path(root, i, j): return os.path.join(root, f"seq_{i:06d}", f"token_{j:05d}.pt")
def load_token(root, i, j): return torch.load(token_path(root, i, j), map_location="cpu", weights_only=True)
def get(d, layer, target):  return d[f"{layer}.{target}"]          # get(d, "L17", "router_logits")

meta   = load_root(root)
layers = [f"L{idx:02d}" for idx in meta["model"]["dumped_layer_indices"]]   # ["L03",...,"L42"]
d      = load_token(root, 0, 0)                                             # seq 0, prefill token
ri     = get(d, "L17", "router_input").float()                             # upcast bf16 → fp32 before math
```

## Must-know facts
- Load with `weights_only=True`. Upcast bf16 (`router_input`, `pre_attn_input`) to fp32 before arithmetic.
- One `.pt` = one sequence, one token; tensors are 1-D, never aggregated.
- `topk_ids` are a **set** of 6 ids — handle order-insensitively.
- A sequence's token files are `token_00000.pt … token_{M-1}.pt` where `M` = number of generated tokens; **enumerate by globbing**, since `M = len(output_token_ids) + 1`. `seq_*` and `token_*` indices may also have gaps (e.g. a skipped engine warmup forward), so always glob rather than assume contiguous indices.
- Layers 0–2 are absent by design; expect only `L03`–`L42`.

## Recipes
**Iterate a sequence**
```python
import glob
tok_files = sorted(glob.glob(os.path.join(root, f"seq_{i:06d}", "token_*.pt")))
for path in tok_files:
    d = torch.load(path, map_location="cpu", weights_only=True)
    # use d["L03.router_logits"], d["L17.router_input"], ...
```
**Routing affinities from logits**
```python
probs = torch.sigmoid(get(d, "L17", "router_logits").float())   # (256,) per-expert affinity
```
**Expert usage histogram over a sequence**
```python
hist = torch.zeros(256, dtype=torch.long)
import glob
for path in sorted(glob.glob(os.path.join(root, f"seq_{i:06d}", "token_*.pt"))):
    d = torch.load(path, map_location="cpu", weights_only=True)
    for L in layers:
        hist += torch.bincount(get(d, L, "topk_ids").long(), minlength=256)
# hist[e] = activations of expert e across all layers/tokens of the sequence
```
**Per-layer activation stats (one token)**
```python
norms = {L: get(d, L, "router_input").float().norm().item() for L in layers}
```
**Which experts missed the LRC cache (decode tokens only)**
```python
# lrc_cache_hit is aligned 1:1 with topk_ids; present only on decode token files.
if any(k.endswith(".lrc_cache_hit") for k in d):           # guard: absent on token_00000 (prefill)
    tk  = get(d, "L17", "topk_ids")                          # (6,) int16, routing order
    hit = get(d, "L17", "lrc_cache_hit")                     # (6,) bool, same order
    misses = [int(tk[i]) for i in range(tk.numel()) if not bool(hit[i])]   # experts paged in on demand
    hit_rate = float(hit.float().mean())                    # fraction of the 6 routed experts already cached
```
**Reproduce which experts were selected (logits + bias)**
```python
m = meta["model"]                                            # from metadata.json
logits = get(d, "L17", "router_logits").float()
bias   = get(d, "L17", "router_bias").float()                # same vector every token of L17
biased = logits.sigmoid() + bias                             # selection ranks on THIS, not on logits
G, kg, k = m["num_expert_group"], m["topk_group"], m["num_experts_per_tok"]
bg = biased.view(G, -1); keep = bg.max(-1).values.topk(kg).indices
mask = torch.zeros(G, dtype=torch.bool); mask[keep] = True
masked = torch.where(mask[:, None], bg, torch.zeros_like(bg)).flatten()
sel = set(masked.topk(k).indices.tolist())                   # == set(get(d,"L17","topk_ids").tolist())
```

## What it supports
Expert load-balancing / co-activation analysis, routing entropy and confidence over `sigmoid(router_logits)`, building `(router_input → topk_ids)` datasets (router probing/distillation), layerwise activation-drift and outlier studies, and comparing any two dumps token-for-token by `(seq_index, token_index)`.

## Gotchas
- `router_logits` are raw scores, not probabilities and not the selection — apply scoring/grouped-topk yourself, or use `topk_ids` for "which experts."
- `lrc_cache_hit` is **decode-only** and **NPU-specific** — `token_00000` (prefill) and CUDA dumps won't have it. Always guard with `any(k.endswith(".lrc_cache_hit") for k in d)` before using it. Unlike `topk_ids`, its order **is** meaningful: element `i` is the hit/miss for `topk_ids[i]`.
- bf16 stores exactly but compute in fp32.
- Token/sequence file paths are derived by convention from indices — there is no separate index file.
