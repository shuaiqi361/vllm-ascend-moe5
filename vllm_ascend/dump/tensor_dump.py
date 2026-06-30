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
        # CHANGE: the mandatory ("core") targets every dumped layer must have for a forward to
        # be written. router_bias is core (always present, every token). lrc_cache_hit (if you
        # applied that guide) is NOT here — it is the optional decode-only target. The integrity
        # gate counts these core keys. (user request: dump the routing selection bias)
        self._core_targets = ("router_input", "pre_attn_input", "router_logits", "topk_ids", "router_bias")
        self.hidden_size = 0
        self.n_routed_experts = 0
        self.num_experts_per_tok = 0
        self._meta = None
        # diagnostics (cheap; only printed when DUMP_DEBUG=1)
        self._n_begin = 0; self._n_written = 0; self._n_discard = 0
        self._f_pre = 0; self._f_route = 0   # per-forward tap-fire counts
        self._dbg_route_shape = True; self._dbg_pre_shape = True  # one-shot shape prints
        self._dbg_bias_shape = True  # NEW: one-shot print for the router_bias capture
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
                  num_hash_layers, num_hidden_layers, scoring_func,
                  num_expert_group=1, topk_group=1, norm_topk_prob=True,
                  routed_scaling_factor=1.0):
        # NEW: called once from the model __init__ with config dims; writes metadata.json.
        # CHANGE: also takes the group-topk + scaling params so the dump is self-describing
        # enough to REPRODUCE the selection from router_logits + router_bias. (user request)
        if not self.enabled or self._cfg_done:
            return
        # NEW: in a distributed run (TP/EP) the model __init__ runs on every rank, so
        # gate writing to global rank 0 only — otherwise N ranks race on the same files.
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
            # n_keys counts the core targets (now 5, incl. router_bias). lrc_cache_hit, if
            # present, is additive/decode-only and excluded from this count.
            self.n_keys = len(tags) * len(self._core_targets)
            self._meta = {
                "platform": os.environ.get("DUMP_PLATFORM", "npu"),
                # CHANGE: record the routing hyperparams needed to reproduce topk_ids offline
                # from router_logits + router_bias (sigmoid -> +bias -> group top-k).
                "model": {"hidden_size": hidden_size, "n_routed_experts": n_routed_experts,
                          "num_experts_per_tok": num_experts_per_tok, "scoring_func": scoring_func,
                          "num_expert_group": num_expert_group, "topk_group": topk_group,
                          "norm_topk_prob": norm_topk_prob, "routed_scaling_factor": routed_scaling_factor,
                          "dumped_layer_indices": list(range(num_hash_layers, num_hidden_layers))},
                "target_order": ["router_input", "pre_attn_input", "router_logits", "topk_ids", "router_bias"],
                # NEW: lrc_cache_hit is registered here only if you also applied that guide; harmless to keep.
                "optional_targets": ["lrc_cache_hit"],
                "targets": {"router_input": {"dim": hidden_size, "dtype": "bfloat16"},
                            "pre_attn_input": {"dim": hidden_size, "dtype": "bfloat16"},
                            "router_logits": {"dim": n_routed_experts, "dtype": "float32"},
                            "topk_ids": {"dim": num_experts_per_tok, "dtype": "int16"},
                            # NEW: the per-expert selection bias added to sigmoid(router_logits).
                            "router_bias": {"dim": n_routed_experts, "dtype": "float32",
                                            "is_parameter": True, "per_token": False,
                                            "semantics": "= gate.e_score_correction_bias. Added to the "
                                                         "sigmoid SCORES (not the logits) for expert "
                                                         "selection via group top-k; routing weights use "
                                                         "the UNBIASED scores. Layer-static (same every token)."},
                            "lrc_cache_hit": {"dim": num_experts_per_tok, "dtype": "bool",
                                              "availability": "decode_tokens_only",
                                              "order": "aligned 1:1 with topk_ids"}},
                "layer_tags": tags,
                "key_scheme": "{layer_tag}.{target}  e.g. 'L03.router_logits'",
                "layout": {
                    "<root>": "= DUMP_DIR, one directory per run",
                    "<root>/metadata.json": "this file: run-level schema (model dims + routing hyperparams, targets, layer tags)",
                    "<root>/seq_{i:06d}/": "one directory per generated sequence (one prompt at --max-num-seqs 1); enumerate by globbing seq_*",
                    "<root>/seq_{i:06d}/seq_meta.json": "that sequence's prompt_token_ids and output_token_ids (len == token-file count - 1)",
                    "<root>/seq_{i:06d}/token_{j:05d}.pt": "one generated token; dict of {layer_tag}.{target} CPU tensors (incl. router_bias every token; lrc_cache_hit on decode only if that guide is applied). load with torch.load(path, map_location='cpu', weights_only=True)",
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
        # ... unchanged above (lock acquire, _in_fwd reset, over-cap early return) ...
        with self._lock:
            self._in_fwd = False
            if not self._capture:
                # Over the per-sequence file cap: nothing was offloaded/recorded; write nothing.
                self._buf = {}
                return
            row = {f"{tag}.{name}": t for tag, d in self._buf.items() for name, t in d.items()}
            self._buf = {}
            # Integrity gate: a correct main-model forward yields all dumped layers with all CORE
            # targets (now 5, including router_bias), each keyed by its real layer id. Anything
            # else (warmup, draft/MTP, a layer whose hooks didn't all fire, a misaligned stamp) is
            # incomplete or mis-attributed -> discard and write nothing.
            # CHANGE: gate on the count of CORE-target keys (== n_keys), not len(row). The optional
            # decode-only lrc_cache_hit (if that guide is applied) adds extra keys on decode tokens,
            # so a total count would exceed n_keys on decode and discard every decode forward.
            n_core = sum(1 for k in row if k.split(".", 1)[1] in self._core_targets)
            if n_core != self.n_keys:
                self._n_discard += 1
                if self._dbg and self._n_discard <= 5:
                    print(f"[DUMP-DEBUG] DISCARD forward (core_keys={n_core}, expected n_keys="
                          f"{self.n_keys}, total_keys={len(row)}): this forward saw pre_attn={self._f_pre} routing={self._f_route}. "
                          f"core==0 -> taps never fired (begin_forward ran but the input_layernorm hook "
                          f"and/or the routing tap did not); 0<core<n_keys -> a tap fired for some "
                          f"layers/targets but not all (check all source edits are applied to THIS build, "
                          f"incl. passing gate.e_score_correction_bias into record_router_io).", flush=True)
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

    def record_router_io(self, hidden_states, router_logits, e_score_correction_bias=None):
        # bf16 router input + fp32 gate logits, captured at the gate site (pre-prepare).
        # CHANGE: also capture the per-expert selection bias (e_score_correction_bias). It is a
        # (n_routed_experts,) fp32 PARAMETER on the gate (NOT a per-token activation): the router
        # adds it to the sigmoid SCORES to choose experts (group top-k), while the routing weights
        # use the UNBIASED scores. So topk_ids depends on router_logits AND this bias. (user request)
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
            # NEW: the bias is a full (n_routed_experts,) vector with NO token dim, so do NOT
            # slice [-1] — store it whole. It is None on hash layers (already excluded above);
            # for MoE layers it is the gate's e_score_correction_bias parameter.
            if e_score_correction_bias is not None:
                b = e_score_correction_bias
                if b.dim() > 1:
                    b = b.squeeze()
                assert b.dim() == 1 and b.shape[0] == self.n_routed_experts, (
                    f"[DUMP] router_bias: {tuple(e_score_correction_bias.shape)} -> sliced {tuple(b.shape)}, "
                    f"expected ({self.n_routed_experts},). Bias tap is at the wrong place or the layout changed.")
                d["router_bias"] = b.detach().to(torch.float32).clone().cpu()
                if self._dbg and self._dbg_bias_shape:
                    self._dbg_bias_shape = False
                    print(f"[DUMP-DEBUG] router_bias tap fired @ layer {layer_idx}: "
                          f"shape={tuple(b.shape)} (fp32 gate.e_score_correction_bias)", flush=True)

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

    def record_cache_hits(self, on_device):
        # NEW: per-routed-expert LRC-cache hit mask for the CURRENT layer, aligned 1:1 with
        # the topk_ids already captured for this layer. on_device = set of expert ids resident
        # in this layer's HBM cache, snapshotted in ExpertOffloadManager._update_weights BEFORE
        # the on-demand load loop. 1 = hit (resident), 0 = miss (about to be paged in).
        # Uses _cur_layer (decoder layer id stamped by the input_layernorm hook) so the tag
        # matches topk_ids — NOT the offload manager's MoE-list layer_idx. Decode path only;
        # prefill bulk-loads and never calls this. (user request)
        if not self.enabled or not self._in_fwd or not self._capture:
            return
        with self._lock:
            layer_idx = self._cur_layer
            if layer_idx < self.num_hash or layer_idx >= self.num_total:
                return
            d = self._buf.get(f"L{layer_idx:02d}")
            if d is None or "topk_ids" not in d:
                # topk for this layer not captured yet (e.g. select_experts tap didn't run
                # before paging) -> skip rather than emit a mask of the wrong order.
                return
            tk = d["topk_ids"]  # (num_experts_per_tok,) int16 cpu, routed-only, in routing order
            try:
                resident = {int(e) for e in on_device}
            except TypeError:
                return
            # bool mask in EXACTLY topk_ids order: hits[i] tells whether topk_ids[i] was resident.
            hits = torch.tensor([1 if int(e) in resident else 0 for e in tk.tolist()],
                                 dtype=torch.bool)
            d["lrc_cache_hit"] = hits
            if self._dbg and self._dbg_cache_shape:
                self._dbg_cache_shape = False
                print(f"[DUMP-DEBUG] lrc_cache_hit tap fired @ layer {layer_idx}: "
                      f"topk={tk.tolist()} hits={hits.tolist()} (1=resident, 0=miss)", flush=True)

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