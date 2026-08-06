#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import signal
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
EVAL_DIR = os.path.join(REPO, "SSCG-Agent开源代码", "evaluation")
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, os.path.dirname(__file__))

# Maximum number of post-processing feedback rounds granted to the enhanced baselines
# (same budget as SCG-Agent's per-checker rewrite limit).
MAX_ROUNDS = 3

from executor_static import ExecutorStaticAgent                              # noqa: E402
from static_analysis_agent import BanditStaticAnalysisTool, CodeQLStaticAnalyzer  # noqa: E402
from functional_test_agent import LLMFunctionalTestAgent, modify_test_code                     # noqa: E402
from executor_agent_safe import execute_fuzz                                 # noqa: E402
from fuzz_agent import InputMutatorAgent, TesterFuzzAgent                    # noqa: E402
from llms import Qwen_LLM                                                    # noqa: E402
import llm_bridge                                                            # noqa: E402

# FT fuzz oracle LLM client
QWEN_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-SET-YOUR-API-KEY")
QWEN_BASE = os.environ.get("DASHSCOPE_BASE_URL", "https://api.example.com/v1")


# ----------------------------------------------------------------------------- LLM
class BridgeLLM:
    """Adapter so the S²CG parsing/programmer prompts can call the baseline's own GPT-4o."""
    def __init__(self, model_key):
        llm_bridge.set_model(model_key)

    def generate(self, prompt):
        return llm_bridge.generate(prompt)


# --------------------------------------------------------------- SA / UT / FT probes
def make_entry(task_id, prompt, test=None):
    return {"ID": task_id, "Prompt": prompt, "test": test}


def probe_sa(entry, code):
    """Returns (passed: bool, feedback_text)."""
    try:
        result, err = ExecutorStaticAgent(entry).execute_static_analysis_gpt(
            code, CodeQLStaticAnalyzer(entry), BanditStaticAnalysisTool(entry))
        passed = result.name == "SAFE"
        return passed, (None if passed else err)   # err = (codeql_issues, bandit_issues)
    except Exception as e:
        return False, f"SA execution error: {e}"


def probe_ut(entry, code):
    """HumanEval only (entry['test'] present). Returns (passed, feedback_text or None)."""
    if not entry.get("test"):
        return None, None   # SecurityEval CWE tasks have no unit tests
    try:
        passed, err = LLMFunctionalTestAgent(entry, None).run_tests(
            code, modify_test_code(entry["test"]), humaneval=True)
        return passed, (None if passed else (err or "unit test failed"))
    except Exception as e:
        return False, f"UT execution error: {e}"


class _Timeout(Exception):
    pass


def _toh(signum, frame):
    raise _Timeout()


def probe_ft(entry, code, qwen_llm, iterations=50):
    """Faithful fuzz loop (mirrors eval_fuzzing.fuzzing) returning (passed, feedback_text)."""
    signal.signal(signal.SIGALRM, _toh)
    try:
        inputs = TesterFuzzAgent(entry, qwen_llm).generate_test_inputs()
    except Exception as e:
        return False, f"FT input-gen error: {e}"
    if not inputs:
        return False, "fuzzing produced no test inputs"
    failed = []
    for _ in range(iterations):
        try:
            signal.alarm(10)
            result, passed, fname = execute_fuzz(code, inputs, 3.0)
        except Exception as e:
            failed.append({"inputs": inputs, "result": str(e)})
            if len(failed) > 3:
                break
            continue
        finally:
            signal.alarm(0)
        if not passed:
            if "No module named" in str(result):
                return False, f"module missing: {result}"
            failed.append({"inputs": inputs, "result": result})
            if len(failed) > 3:
                break
        try:
            inputs = InputMutatorAgent(inputs, code, fname).mutate_inputs()
        except Exception:
            break
    if failed:
        return False, "fuzzing failures:\n" + json.dumps(failed[:3], ensure_ascii=False, default=str)
    return True, None


# --------------------------------------------------------------- feedback + repair
def summarize_sa(llm, sa_err):
    """Port of parsing_agent.extract_static_analysis_results (verbatim prompt)."""
    res = str(sa_err[0]) + "\n" + str(sa_err[1])
    prompt = (
        "Please review the static analysis output below for a piece of code:\n"
        f"{res}\n"
        "In a concise manner, summarize the issues identified in the analysis and "
        "provide recommended fixes for each issue. Use clear and succinct language.\n"
        "*Issues*:\n*Recommended fixes*:\n"
    )
    try:
        return llm.generate(prompt)
    except Exception as e:
        return f"(SA summary failed: {e})\n{res}"


def build_feedback(llm, sa, ut, ft):
    """Combine whatever failed into a single merged feedback block (three-checker full feedback)."""
    parts = []
    if sa is not None:
        parts.append("### Static Analysis (security)\n" + summarize_sa(llm, sa))
    if ut is not None:
        parts.append("### Unit Tests (functionality)\n" + str(ut))
    if ft is not None:
        parts.append("### Fuzz Testing (robustness)\n" + str(ft))
    return "\n\n".join(parts)


def repair_once(llm, code, feedback):
    """Port of programmer_agent.call_chatgpt_programmer_feedback_static (verbatim prompt)."""
    prompt = (
        "Please analyze the original code provided below:\n"
        f"```python\n{code}\n```\n"
        "Based on the issues and recommended fixes described below:\n"
        f"{feedback}\n"
        "# Follow the steps below:\n"
        "Step 1. Think step by step in plain text (no code blocks), explain the reason for the issues.\n"
        "Step 2. Reason about how to fix it.\n"
        "Step 3. At the end, output the final fixed code wrapped in ```python```.\n"
    )
    try:
        out = llm.generate(prompt).strip()
        m = re.findall(r"```python\s*(.*?)```", out, re.DOTALL)
        return m[-1] if m else ""
    except Exception as e:
        print(f"[repair_once] error: {e}")
        return ""


# ----------------------------------------------------------------------------- main
def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="self_collaboration")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--ft-iterations", type=int, default=50)
    ap.add_argument("--skip-ft", action="store_true", help="skip FT probe; rejudge reuses original eval_summary FT verdict")
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", dest="outp", default=None)
    args = ap.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    inp = args.inp or os.path.join(results_dir, args.baseline, args.model, "result.jsonl")
    out_dir = args.outp or os.path.join(results_dir, f"{args.baseline}_pp", args.model)
    os.makedirs(out_dir, exist_ok=True)

    # eval_summary -> per-checker fail_id sets (the incremental partition)
    summary_path = os.path.join(REPO, "major_revision_results", "eval_summary.json")
    es = json.load(open(summary_path, encoding="utf-8"))[f"{args.baseline}_{args.model}"]
    recs = load_jsonl(inp)
    # fail_ids are task_id strings (e.g. 'HumanEval/9', 'CWE-020_codeql_1.py'), NOT row indices;
    # map them to row indices so the per-row loop below matches correctly.
    tid2idx = {r["task_id"]: i for i, r in enumerate(recs)}
    def _to_idx(idset):
        out = set()
        for x in idset:
            if isinstance(x, int):
                out.add(x)
            elif x in tid2idx:
                out.add(tid2idx[x])
            else:
                print(f"  [warn] fail_id not in result.jsonl: {x}")
        return out
    sa_fail = _to_idx(es["static_analysis"]["fail_ids"])
    ut_fail = _to_idx(es["unit_test"]["fail_ids"])
    ft_fail = _to_idx(es["fuzzing"]["fail_ids"])
    orig_ft_fail = set(ft_fail)          # original FT verdict, reused when --skip-ft
    if args.skip_ft:
        ft_fail = set()                  # do not probe FT
        print("[--skip-ft] FT probe disabled; FT verdicts reused from eval_summary")
    to_fix_idx = sa_fail | ut_fail | ft_fail
    he = {r["task_id"]: r for r in load_jsonl(os.path.join(EVAL_DIR, os.pardir, "humaneval.jsonl"))}
    # SecurityEval tasks carry no `test`; HumanEval tasks do.
    llm = BridgeLLM(args.model)
    fuzz_llm = BridgeLLM(args.model)   # FT fuzz oracle

    enhanced = {"sa": [], "ut": [], "ft": [], "fail_ids": {"sa": [], "ut": [], "ft": []}}
    out_recs, log_recs = [], []
    n_untouched = n_repaired = n_sa_ok = n_ut_ok = n_ft_ok = 0

    for i, rec in enumerate(recs):
        tid, prompt, code = rec["task_id"], rec.get("prompt") or rec.get("Prompt"), rec["code"]
        test = he.get(tid, {}).get("test") if tid in he else None
        entry = make_entry(tid, prompt, test)

        if i not in to_fix_idx:
            # untouched: code unchanged -> keep original (all-3-pass) verdict
            n_untouched += 1
            enhanced["sa"].append(True); enhanced["ut"].append(test is not None)
            enhanced["ft"].append(True)
            out_recs.append({"task_id": tid, "code": code, "api_calls": rec.get("api_calls", 0)})
            log_recs.append({"idx": i, "task_id": tid, "untouched": True})
            continue

        # to_fix: iteratively repair for up to MAX_ROUNDS rounds. Each round re-probes
        # the three checkers on the CURRENT code; if any fail, combine their feedback
        # and revise once; stop early as soon as all three pass.
        rounds_used = 0
        for round_idx in range(MAX_ROUNDS):
            rounds_used = round_idx + 1
            sa_pass, sa_fb = probe_sa(entry, code)
            ut_pass, ut_fb = probe_ut(entry, code) if test else (None, None)
            if args.skip_ft:
                ft_pass = (i not in orig_ft_fail)
                ft_fb = None
            else:
                ft_pass, ft_fb = probe_ft(entry, code, fuzz_llm, args.ft_iterations)
            feedback = build_feedback(llm,
                                      None if sa_pass else sa_fb,
                                      None if (ut_pass is None or ut_pass) else ut_fb,
                                      None if ft_pass else ft_fb)
            if not feedback:
                break  # all three pass -> no further revision needed
            new_code = repair_once(llm, code, feedback)
            if new_code.strip():
                code = new_code
        n_repaired += 1

        n_sa_ok += int(sa_pass); n_ut_ok += int(ut_pass is True); n_ft_ok += int(ft_pass)
        if not sa_pass:
            enhanced["fail_ids"]["sa"].append(i)
        if ut_pass is False:
            enhanced["fail_ids"]["ut"].append(i)
        if not ft_pass:
            enhanced["fail_ids"]["ft"].append(i)
        enhanced["sa"].append(sa_pass); enhanced["ut"].append(ut_pass is True)
        enhanced["ft"].append(ft_pass)
        out_recs.append({"task_id": tid, "code": code, "api_calls": rec.get("api_calls", 0)})
        log_recs.append({"idx": i, "task_id": tid, "untouched": False,
                         "sa": sa_pass, "ut": ut_pass, "ft": ft_pass, "rounds": rounds_used})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(recs)}] untouched={n_untouched} repaired={n_repaired} "
                  f"(post sa_ok={n_sa_ok} ut_ok={n_ut_ok} ft_ok={n_ft_ok})")

    ut_total = sum(1 for r in recs if (he.get(r["task_id"]) or {}).get("test"))
    summary = {
        "sa_pass@1": round(100 * sum(enhanced["sa"]) / 285, 2),
        "ut_pass@1": round(100 * sum(enhanced["ut"]) / ut_total, 2) if ut_total else None,
        "ft_pass@1": round(100 * sum(enhanced["ft"]) / 285, 2),
        "sa_success": sum(enhanced["sa"]), "ft_success": sum(enhanced["ft"]),
        "ut_success": sum(enhanced["ut"]), "ut_total": ut_total,
        "untouched": n_untouched, "repaired": n_repaired,
        "max_feedback_rounds": MAX_ROUNDS,
        "fail_ids": enhanced["fail_ids"],
    }
    json.dump(summary, open(os.path.join(out_dir, "enhanced_summary.json"), "w"),
              indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "result.jsonl"), "w", encoding="utf-8") as f:
        for r in out_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "pp_log.jsonl"), "w", encoding="utf-8") as f:
        for r in log_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[done] {out_dir}")
    print(f"  Enhanced {args.baseline} ({args.model}): "
          f"SA {summary['sa_pass@1']}%  FT {summary['ft_pass@1']}%  "
          f"UT {summary['ut_pass@1']}%")
    print(f"  untouched(=original)={n_untouched}  repaired={n_repaired}")
    print("  vs original (eval_summary): SA %.2f%% FT %.2f%% UT %.2f%%"
          % (100*es['static_analysis']['success']/285,
             100*es['fuzzing']['success']/285,
             100*es['unit_test']['success']/164))
    print("  (optional full re-eval)  python major_revision_results/run_scg_eval.py " + out_dir)


if __name__ == "__main__":
    main()
