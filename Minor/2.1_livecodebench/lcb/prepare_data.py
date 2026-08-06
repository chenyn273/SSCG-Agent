"""Prepare the LiveCodeBench subset for the leakage-free supplementary experiment.

Loads ``livecodebench/code_generation_lite`` from HuggingFace, keeps problems
whose ``contest_date`` is on/after the cutoff (default 2024-08-01, i.e. after
the latest LLM training cutoff per ticket 03), prefers competitive
stdin/stdout problems (empty ``starter_code``), and writes a JSONL the rest of
the pipeline consumes.

The exact field names of ``code_generation_lite`` are discovered at runtime
(see ``--inspect``): this script prints the first record's keys and a sample so
the schema can be confirmed on the server before trusting the export.

Output record schema:
    {
      "task_id": "<question_id>",
      "question_id": "...",
      "contest_date": "YYYY-MM-DD",
      "is_stdin": True/False,
      "prompt": "<competitive-programming framed problem statement>",
      "public_inputs":  [...],
      "public_outputs": [...],
      "private_inputs":  [...],   # hidden tests, when present (lite prunes these)
      "private_outputs": [...],
    }
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date

CUTOFF = date(2024, 8, 1)

CP_PREAMBLE = (
    "Please solve the following competitive programming problem. "
    "Write a complete, runnable Python 3 program that reads input from standard input (stdin) "
    "and writes the result to standard output (stdout). "
    "Do not write any explanation, tests, or example usage; output only the program.\n\n"
    "Problem:\n"
)


def _parse_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def _as_list(x):
    """Normalise a test-case field into a list[str] (or list of lines)."""
    if x is None:
        return []
    if isinstance(x, list):
        return [str(e) for e in x]
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        # {'input': [...], 'output': [...]} shape
        return _as_list(x.get("input") or x.get("inputs"))
    return [str(x)]


def _decode_tests_string(s: str):
    """LiveCodeBench stores test cases as a JSON string (public) or a
    base64+zlib-compressed JSON string (private). Decode to a python object."""
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        import base64
        import zlib
        raw = zlib.decompress(base64.b64decode(s))
        return json.loads(raw.decode("utf-8", "ignore"))
    except Exception:
        return None


def extract_tests(rec):
    """Pull (public_inputs, public_outputs, private_inputs, private_outputs) from a record.

    LiveCodeBench has shipped several schemas over releases; we handle the
    common ones defensively and log what we actually found.
    """
    pub_in, pub_out, priv_in, priv_out = [], [], [], []

    # public_tests can be dict {input,output} or a JSON string, or absent.
    def split_io(field_val):
        ins, outs = [], []
        if field_val is None:
            return ins, outs
        if isinstance(field_val, str):
            decoded = _decode_tests_string(field_val)
            if decoded is None:
                return _as_list(field_val), []
            field_val = decoded
        if isinstance(field_val, dict):
            ins = _as_list(field_val.get("input") or field_val.get("inputs"))
            outs = _as_list(field_val.get("output") or field_val.get("outputs"))
        elif isinstance(field_val, list):
            # could be list of {'input':..,'output':..} pairs
            if field_val and isinstance(field_val[0], dict):
                for pair in field_val:
                    ins += _as_list(pair.get("input"))
                    outs += _as_list(pair.get("output"))
            else:
                ins = _as_list(field_val)
        return ins, outs

    for key in ("public_tests", "public_test_cases", "publicTests"):
        if key in rec:
            pub_in, pub_out = split_io(rec[key])
            break
    for key in ("private_tests", "private_test_cases", "privateTests"):
        if key in rec:
            priv_in, priv_out = split_io(rec[key])
            break

    return pub_in, pub_out, priv_in, priv_out


def frame_prompt(question_content: str) -> str:
    return CP_PREAMBLE + (question_content or "").strip() + "\n"


def build_record(rec, idx: int):
    qid = rec.get("question_id") or rec.get("id") or f"lcb_{idx}"
    cdate = rec.get("contest_date")
    starter = rec.get("starter_code") or ""
    is_stdin = not str(starter).strip()  # empty starter => stdin/stdout problem
    pub_in, pub_out, priv_in, priv_out = extract_tests(rec)
    question = rec.get("question_content") or rec.get("question") or rec.get("prompt") or ""
    return {
        "task_id": str(qid),
        "question_id": str(qid),
        "contest_date": str(cdate) if cdate is not None else "",
        "is_stdin": bool(is_stdin),
        "prompt": frame_prompt(question),
        "question_content": question,
        "public_inputs": pub_in,
        "public_outputs": pub_out,
        "private_inputs": priv_in,
        "private_outputs": priv_out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="livecodebench/code_generation_lite")
    ap.add_argument("--config_name", default="release_v6",
                    help="HF config/release (default release_v6 = 2024-10..2025-05, all >= cutoff)")
    ap.add_argument("--start_date", default="2024-08-01")
    ap.add_argument("--prefer_stdin", action="store_true", default=True)
    ap.add_argument("--stdin_only", action="store_true", help="keep only stdin problems")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--inspect", action="store_true", help="print schema of first record and exit")
    ap.add_argument("--local_jsonl", default=None,
                    help="comma/glob list of raw LCB jsonl files to read directly (skip HF datasets load)")
    args = ap.parse_args()

    from datasets import load_dataset

    cutoff = _parse_date(args.start_date) or CUTOFF

    if args.local_jsonl:
        import glob as _g
        paths = []
        for tok in args.local_jsonl.split(","):
            paths.extend(_g.glob(tok.strip()))
        rows = []
        for p in sorted(paths):
            print(f"reading local jsonl: {p}")
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        print(f"Loaded {len(rows)} rows from local jsonl ({len(paths)} files). "
              f"First-row keys: {list(rows[0].keys()) if rows else []}")
    else:
        print(f"Loading {args.dataset} (config={args.config_name}) ...")
        if args.config_name:
            ds = load_dataset(args.dataset, args.config_name, split="test", trust_remote_code=True)
        else:
            ds = load_dataset(args.dataset, split="test", trust_remote_code=True)
        rows = list(ds)
        print(f"Loaded {len(rows)} rows. First-row keys: {list(rows[0].keys())}")

    if args.inspect:
        print("\n=== FIRST RECORD ===")
        for k, v in rows[0].items():
            sval = repr(v)
            if len(sval) > 600:
                sval = sval[:600] + " ...<truncated>"
            print(f"  {k}: {sval}")
        print("\n=== dates sample ===")
        print([r.get("contest_date") for r in rows[:5]])
        print("min/max contest_date present:",
              min((r.get("contest_date") or "") for r in rows),
              max((r.get("contest_date") or "") for r in rows))
        return

    kept = []
    stdin_n, fn_n, before = 0, 0, 0
    for idx, rec in enumerate(rows):
        d = _parse_date(rec.get("contest_date"))
        if d is None or d < cutoff:
            before += 1
            continue
        rec_out = build_record(rec, idx)
        if rec_out["is_stdin"]:
            stdin_n += 1
        else:
            fn_n += 1
            if args.stdin_only:
                continue
        kept.append(rec_out)

    print(f"After cutoff {cutoff}: {len(kept)} kept ({stdin_n} stdin, {fn_n} function-style), {before} before cutoff.")
    if args.prefer_stdin and not args.stdin_only:
        kept.sort(key=lambda r: (not r["is_stdin"]))  # stdin first, stable
        print("Reordered: stdin problems first.")

    if args.limit:
        kept = kept[: args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(kept)} problems -> {args.output}")
    if kept:
        r0 = kept[0]
        print("Sample task_id:", r0["task_id"], "| date:", r0["contest_date"],
              "| stdin:", r0["is_stdin"], "| #pub_tests:", len(r0["public_inputs"]),
              "| #priv_tests:", len(r0["private_inputs"]))


if __name__ == "__main__":
    main()
