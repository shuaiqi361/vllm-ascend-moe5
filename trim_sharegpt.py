#!/usr/bin/env python3
"""
Trim ShareGPT prompts to the first N characters.

ShareGPT_V3_unfiltered_cleaned_split.json layout
-------------------------------------------------
[
  {
    "id": "<string>",
    "conversations": [
      {"from": "human",  "value": "<prompt text>"},
      {"from": "gpt",    "value": "<response text>"},
      ...
    ]
  },
  ...
]

This script truncates ONLY the `value` of turns whose `from` role is in
ROLES_TO_TRIM (default: just "human", i.e. the prompts). Every other field
-- the `id`, the `from` role, and the `gpt`/`system` values -- is left byte
for byte unchanged. A value shorter than N is left as-is.

Output is written next to the input as <name>_trim_<N>.json
e.g. ShareGPT_V3_unfiltered_cleaned_split.json
  -> ShareGPT_V3_unfiltered_cleaned_split_trim_1024.json
"""

import argparse
import json
import os
import sys

# Which turn roles count as "prompts" and get trimmed.
#   {"human"}            -> only human prompts          (default; matches the task)
#   {"human", "system"}  -> human + any system context
#   {"human","gpt","system"} -> trim everything
# vLLM's native ShareGPT path actually only uses conversations[0] as the prompt;
# trimming all human turns is a strict superset of that and is harmless.
ROLES_TO_TRIM = {"human"}


def trim_value(value, n):
    """Return value truncated to the first n characters, if it's a long string."""
    if isinstance(value, str) and len(value) > n:
        return value[:n], True
    return value, False


def trim_dataset(data, n):
    """Trim prompts in place. Returns (turns_seen, turns_trimmed)."""
    seen = 0
    trimmed = 0
    for entry in data:
        # Be defensive: skip anything that doesn't look like a conversation entry.
        if not isinstance(entry, dict):
            continue
        conversations = entry.get("conversations")
        if not isinstance(conversations, list):
            continue
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            if turn.get("from") in ROLES_TO_TRIM:
                seen += 1
                new_value, was_trimmed = trim_value(turn.get("value"), n)
                if was_trimmed:
                    turn["value"] = new_value
                    trimmed += 1
    return seen, trimmed


def derive_output_path(input_path, n):
    base, ext = os.path.splitext(input_path)
    return f"{base}_trim_{n}{ext}"


def main():
    parser = argparse.ArgumentParser(
        description="Trim ShareGPT prompts (human turns) to the first N characters."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="ShareGPT_V3_unfiltered_cleaned_split.json",
        help="Path to the input ShareGPT json (default: %(default)s in the cwd).",
    )
    parser.add_argument(
        "-n",
        "--num-chars",
        type=int,
        default=1024,
        help="Max characters to keep per prompt (default: %(default)s).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path (default: <input>_trim_<N>.json next to the input).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"error: input file not found: {args.input}")
    if args.num_chars < 0:
        sys.exit("error: -n/--num-chars must be >= 0")

    output_path = args.output or derive_output_path(args.input, args.num_chars)

    print(f"Loading {args.input} ...", flush=True)
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        sys.exit("error: expected the top-level JSON to be a list of conversations")

    print(f"Trimming '{'/'.join(sorted(ROLES_TO_TRIM))}' turns to {args.num_chars} chars ...",
          flush=True)
    seen, trimmed = trim_dataset(data, args.num_chars)

    print(f"Writing {output_path} ...", flush=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # ensure_ascii=False keeps non-ASCII text intact (no \uXXXX bloat);
        # compact separators keep the file size close to the minified original.
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    print(
        f"Done. {len(data)} conversations | {seen} prompt turns | "
        f"{trimmed} trimmed (>{args.num_chars} chars) | "
        f"{seen - trimmed} already short enough."
    )


if __name__ == "__main__":
    main()