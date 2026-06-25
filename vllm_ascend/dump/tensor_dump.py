# NEW FILE — tensor-dump driver for DeepSeek-V4 routed-MoE activations.
# Toggle: env DUMP=1 enables; anything else => every method early-returns (no overhead).
# Output dir: env DUMP_DIR (default ./dumps). Platform tag: env DUMP_PLATFORM (default "npu").
# Captures per generated token, per non-hash layer (skips the first num_hash_layers):
#   router_input (bf16,Dh), pre_attn_input (bf16,Dh), router_logits (fp32,E), topk_ids (int16,top_k)
# Token 0 of a sequence = prefill (last prompt position); tokens 1.. = decode steps.
# Designed for `vllm serve ... --enforce-eager --max-num-seqs 1` (see guide).
import os
import json
import threading

import torch


class _Dumper:
    def __init__(self):
        # NEW: master switch. DUMP=1 -> active; else fully inert.
        self.enabled = os.environ.get("DUMP", "0") == "1"
        self.dir = os.environ.get("DUMP_DIR", "./dumps")
        self._dbg = os.environ.get("DUMP_DEBUG", "0") == "1"  # 1 -> print why forwards write/discard
        # Hard cap on files saved per sequence (token_00000..token_{max-1}); 0 = unlimited.
        # token_0 = prefill, then decodes; the bench script sets this from OUTPUT_LEN, so
        # a sequence yields exactly OUTPUT_LEN files (1 prefill + OUTPUT_LEN-1 decode).
        # Beyond the cap, the forward is NOT offloaded from NPU and NOT written.
        try:
            self.max_tokens = int(os.environ.get("DUMP_MAX_TOKENS", "0") or "0")
        except ValueError:
            self.max_tokens = 0
        self._cfg_done = False
        self._lock = threading.RLock()  # serve runs one forward at a time; lock guards state anyway
        # config (set by configure())
        self.num_hash = 0
        self.num_total = 0
        self.n_keys = 0
        self.hidden_size = 0
        self.n_routed_experts = 0
        self.num_experts_per_tok = 0
        self._meta = None
        # diagnostics (cheap; only printed when DUMP_DEBUG=1)
        self._n_begin = 0; self._n_written = 0; self._n_discard = 0
        self._f_pre = 0; self._f_route = 0   # per-forward tap-fire counts
        self._dbg_route_shape = True; self._dbg_pre_shape = True  # one-shot shape prints
        # per-forward state
        self._in_fwd = False
        self._capture = True      # False -> over the per-sequence cap: skip offload + write
        self._cur_layer = -1      # actual layer_idx of the layer currently executing (stamped by mark_pre_attn)
        self._buf = {}            # layer_tag -> {target_name: 1-D cpu tensor}
        self._is_prefill = False
        self._pending_out = None  # decode input token id, committed only if the forward is written
        # per-sequence state
        self._decoding = False    # flips True at a sequence's first decode forward
        self._seq = -1
        self._tok = 0             # last written decode token index (0 reserved for prefill)
        self._prompt_ids = []
        self._output_ids = []
        if self.enabled:
            os.makedirs(self.dir, exist_ok=True)

    def configure(self, *, hidden_size, n_routed_experts, num_experts_per_tok,
                  num_hash_layers, num_hidden_layers, scoring_func):
        # NEW: called once from the model __init__ with config dims; writes metadata.json.
        if not self.enabled or self._cfg_done:
            return
        # NEW: in a distributed run (TP/EP) the model __init__ runs on every rank, so
        # gate writing to global rank 0 only — otherwise N ranks race on the same files.
        # Requires PP=1 (rank 0 holds all layers) and EP=1 (rank 0 sees the full,
        # unsplit token set at select_experts). See Assumptions.
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                self.enabled = False
                return
        except Exception:
            pass
        with self._lock:
            self.num_hash = num_hash_layers
            self.num_total = num_hidden_layers
            self.hidden_size = hidden_size
            self.n_routed_experts = n_routed_experts
            self.num_experts_per_tok = num_experts_per_tok
            tags = [f"L{i:02d}" for i in range(num_hash_layers, num_hidden_layers)]
            self.n_keys = len(tags) * 4
            self._meta = {
                "platform": os.environ.get("DUMP_PLATFORM", "npu"),
                "model": {"hidden_size": hidden_size, "n_routed_experts": n_routed_experts,
                          "num_experts_per_tok": num_experts_per_tok, "scoring_func": scoring_func,
                          "dumped_layer_indices": list(range(num_hash_layers, num_hidden_layers))},
                "target_order": ["router_input", "pre_attn_input", "router_logits", "topk_ids"],
                "targets": {"router_input": {"dim": hidden_size, "dtype": "bfloat16"},
                            "pre_attn_input": {"dim": hidden_size, "dtype": "bfloat16"},
                            "router_logits": {"dim": n_routed_experts, "dtype": "float32"},
                            "topk_ids": {"dim": num_experts_per_tok, "dtype": "int16"}},
                "layer_tags": tags,
                "key_scheme": "{layer_tag}.{target}  e.g. 'L03.router_logits'",
                "layout": {
                    "<root>": "= DUMP_DIR, one directory per run",
                    "<root>/metadata.json": "this file: run-level schema (model dims, the 4 targets, layer tags)",
                    "<root>/seq_{i:06d}/": "one directory per generated sequence (one prompt at --max-num-seqs 1); enumerate by globbing seq_*",
                    "<root>/seq_{i:06d}/seq_meta.json": "that sequence's prompt_token_ids and output_token_ids (len == token-file count - 1)",
                    "<root>/seq_{i:06d}/token_{j:05d}.pt": "one generated token (token_00000 = prefill/first token, token_k = decode k); a dict of {layer_tag}.{target} CPU tensors; load with torch.load(path, map_location='cpu', weights_only=True)",
                },
                "file_format": {"serialization": "torch.save",
                                "load": "torch.load(path, map_location='cpu', weights_only=True)"},
                "dump_options": {"max_tokens_per_seq": self.max_tokens if self.max_tokens > 0 else "unlimited"},
            }
            self._write_json(os.path.join(self.dir, "metadata.json"), self._meta)
            self._cfg_done = True
            if self._dbg:
                _cap = self.max_tokens if self.max_tokens > 0 else "unlimited"
                print(f"[DUMP-DEBUG] configured: dir={self.dir} dumped_layers={len(tags)} "
                      f"n_keys={self.n_keys} max_tokens_per_seq={_cap} (rank0); waiting for forwards", flush=True)

    # ---- forward lifecycle: driven by hooks on DeepseekV4Model ----
    def begin_forward(self, input_ids, positions):
        # NEW: start of each main-model forward. With --max-num-seqs 1 each forward is
        # one sequence. Classifies by POSITION (not token count): position 0 == a fresh
        # prefill == new sequence. Requires --no-enable-prefix-caching so later prompts
        # also start at position 0. Chunked prefill (always on in V1) is handled: a
        # non-zero-position multi-token forward before any decode is a prefill chunk.
        if not self.enabled or not self._cfg_done:
            return
        try:
            # Best-effort skip of the engine's startup profiling/dummy run so it
            # doesn't create a bogus sequence. Attribute may not exist on all
            # versions; if absent we proceed (see Assumptions for the caveat).
            from vllm.forward_context import get_forward_context
            if getattr(get_forward_context(), "in_profile_run", False):
                return
        except Exception:
            pass
        with self._lock:
            ids = input_ids.flatten()
            pos0 = int(positions.flatten()[0].item())
            n_tok = ids.numel()
            if pos0 == 0:
                # Fresh prefill (first chunk) => new sequence. No filesystem op here;
                # the directory is created lazily on the first successful write, so a
                # forward that gets discarded leaves no empty/partial folder.
                self._seq += 1
                self._tok = 0
                self._decoding = False
                self._prompt_ids = ids.tolist()
                self._output_ids = []
                self._is_prefill = True
                self._pending_out = None
            elif not self._decoding and n_tok > 1:
                # Chunked-prefill continuation chunk: still the prompt, not a decode.
                self._prompt_ids.extend(ids.tolist())
                self._is_prefill = True
                self._pending_out = None
            else:
                # Decode step. Its input token == the token generated previously.
                # Stash it; commit to output_ids only if this forward is actually written.
                self._decoding = True
                self._is_prefill = False
                self._pending_out = int(ids[-1].item())
            self._in_fwd = True
            self._cur_layer = -1
            self._buf = {}
            self._f_pre = 0
            self._f_route = 0
            # Per-sequence file cap: the index this forward would write is 0 for prefill,
            # else _tok+1. Capture only while that index is below max_tokens. Beyond the
            # cap the taps skip (no NPU->host offload) and end_forward writes nothing.
            if self.max_tokens > 0:
                _next_idx = 0 if self._is_prefill else (self._tok + 1)
                self._capture = _next_idx < self.max_tokens
            else:
                self._capture = True
            self._n_begin += 1
            if self._dbg and self._n_begin <= 3:
                print(f"[DUMP-DEBUG] begin_forward #{self._n_begin}: n_tok={n_tok} pos0={pos0} "
                      f"is_prefill={self._is_prefill} capture={self._capture}", flush=True)

    def end_forward(self):
        # NEW: end of each main-model forward. Writes the token immediately (prefill ->
        # token 0, decode -> next index) plus seq_meta.json, so all data is durable on
        # disk without relying on atexit (the EngineCore subprocess may be hard-killed).
        if not self.enabled or not self._in_fwd:
            return
        with self._lock:
            self._in_fwd = False
            if not self._capture:
                # Over the per-sequence file cap: nothing was offloaded/recorded; write nothing.
                self._buf = {}
                return
            row = {f"{tag}.{name}": t for tag, d in self._buf.items() for name, t in d.items()}
            self._buf = {}
            # Integrity gate: a correct main-model forward yields exactly n_keys =
            # (num_total - num_hash) * 4 entries, i.e. all dumped layers with all 4
            # targets, each keyed by its real layer id. Anything else (warmup, draft/MTP,
            # a layer whose hooks didn't all fire, a misaligned stamp) is incomplete or
            # mis-attributed -> discard and write nothing. With <=4 keys per tag and only
            # 40 possible tags, len(row)==n_keys can ONLY be all 40 layers x 4 targets,
            # so this guarantees correct per-layer attribution in every written file.
            if len(row) != self.n_keys:
                self._n_discard += 1
                if self._dbg and self._n_discard <= 5:
                    print(f"[DUMP-DEBUG] DISCARD forward (row_keys={len(row)}, expected n_keys="
                          f"{self.n_keys}): this forward saw pre_attn={self._f_pre} routing={self._f_route}. "
                          f"keys==0 -> taps never fired (begin_forward ran but the input_layernorm hook "
                          f"and/or select_experts tap did not); keys>0 but <n_keys -> one tap fired, the "
                          f"other did not (check both source edits are applied to THIS build).", flush=True)
                return
            os.makedirs(self._seq_dir(), exist_ok=True)  # lazy: only for written forwards
            if self._is_prefill:
                # Prefill => token 0. Single forward normally; under chunked prefill the
                # final chunk's write wins (atomic overwrite).
                self._write_token(0, row)
            else:
                self._tok += 1
                self._write_token(self._tok, row)
                if self._pending_out is not None:
                    self._output_ids.append(self._pending_out)
                    self._pending_out = None
            self._n_written += 1
            if self._dbg and self._n_written <= 3:
                print(f"[DUMP-DEBUG] WROTE {'token_00000 (prefill)' if self._is_prefill else f'token_{self._tok:05d} (decode)'} "
                      f"to seq_{self._seq:06d} (row_keys={len(row)})", flush=True)
            # Rewrite seq_meta each written forward so it is always current and durable.
            self._write_json(os.path.join(self._seq_dir(), "seq_meta.json"),
                             {"seq_index": self._seq, "prompt_token_ids": self._prompt_ids,
                              "output_token_ids": self._output_ids})

    # ---- taps ----
    def _slice1d(self, t, expected_len, name):
        # Single batch, single sequence: take the last token's row, drop any size-1
        # dims, and assert it is exactly 1-D of the expected length before we offload.
        v = t[-1]
        if v.dim() > 1:
            v = v.squeeze()
        assert v.dim() == 1 and v.shape[0] == expected_len, (
            f"[DUMP] {name}: tensor {tuple(t.shape)} -> sliced {tuple(v.shape)}, "
            f"expected ({expected_len},). Tap is at the wrong place or the layout changed.")
        return v

    def record_router_io(self, hidden_states, router_logits):
        # bf16 router input + fp32 gate logits, captured at the gate site (pre-prepare).
        if not self.enabled or not self._in_fwd or not self._capture:
            return
        with self._lock:
            layer_idx = self._cur_layer
            if layer_idx < self.num_hash or layer_idx >= self.num_total:
                return
            ri = self._slice1d(hidden_states, self.hidden_size, "router_input")
            rl = self._slice1d(router_logits, self.n_routed_experts, "router_logits")
            d = self._buf.setdefault(f"L{layer_idx:02d}", {})
            d["router_input"] = ri.detach().to(torch.bfloat16).clone().cpu()
            d["router_logits"] = rl.detach().to(torch.float32).clone().cpu()

    def record_topk(self, topk_ids):
        # routed-only selected expert ids, captured in select_experts (its only source).
        if not self.enabled or not self._in_fwd or not self._capture:
            return
        with self._lock:
            layer_idx = self._cur_layer
            if layer_idx < self.num_hash or layer_idx >= self.num_total:
                return
            tk = self._slice1d(topk_ids, self.num_experts_per_tok, "topk_ids")
            self._buf.setdefault(f"L{layer_idx:02d}", {})["topk_ids"] = (
                tk.detach().to(torch.int16).clone().cpu())

    def mark_pre_attn(self, layer_idx, x):
        # NEW: called from a forward hook on each non-hash layer's input_layernorm, which
        # runs before that layer's attention and MoE. (1) stamps the current layer id (the
        # actual layer_idx) so record_routing attributes the MoE outputs to THIS layer;
        # (2) saves pre_attn_input. Both keyed by the real layer id, in execution order.
        if not self.enabled or not self._in_fwd or not self._capture:
            return
        with self._lock:
            self._cur_layer = layer_idx
            if self._dbg and self._dbg_pre_shape:
                self._dbg_pre_shape = False
                print(f"[DUMP-DEBUG] pre_attn tap fired @ layer {layer_idx}: input_layernorm output shape="
                      f"{tuple(x.shape)} -> stored slice {tuple(self._slice1d(x, self.hidden_size, 'pre_attn_input').shape)}",
                      flush=True)
            pa = self._slice1d(x, self.hidden_size, "pre_attn_input")
            self._buf.setdefault(f"L{layer_idx:02d}", {})["pre_attn_input"] = (
                pa.detach().to(torch.bfloat16).clone().cpu())
            self._f_pre += 1

    # ---- io helpers ----
    def _seq_dir(self):
        return os.path.join(self.dir, f"seq_{self._seq:06d}")

    def _safe_replace(self, tmp, path):
        try:
            os.replace(tmp, path)
            return True
        except OSError as e:
            self._disable_on_error(e, path)
            try: os.remove(tmp)
            except OSError: pass
            return False

    def _disable_on_error(self, e, path):
        if self.enabled:
            print(f"[DUMP-ERR] write failed for {path} ({e.__class__.__name__}: {e}); "
                  f"disabling dump for the rest of the run (engine keeps serving). "
                  f"Free space in {self.dir}, or lower OUTPUT_LEN / number of prompts.", flush=True)
        self.enabled = False

    def _write_token(self, j, row):
        path = os.path.join(self._seq_dir(), f"token_{j:05d}.pt")
        try:
            torch.save(row, path + ".tmp")
        except OSError as e:
            self._disable_on_error(e, path)
            try: os.remove(path + ".tmp")
            except OSError: pass
            return
        self._safe_replace(path + ".tmp", path)

    def _write_json(self, path, obj):
        try:
            with open(path + ".tmp", "w") as f:
                json.dump(obj, f, indent=2)
        except OSError as e:
            self._disable_on_error(e, path)
            return
        self._safe_replace(path + ".tmp", path)


# NEW: process-wide singleton imported by the tap and the wiring.
DUMPER = _Dumper()