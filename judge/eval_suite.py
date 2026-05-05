"""
Unified evaluation entry for steering results (subset filters + lmdeploy judge).

Defaults match this repo layout:
  - ViP-Bench: bbox split only (same as behaviors.VIP_BENCH_QUESTIONS_JSONL).
  - BLINK MC: only Relative_Reflectance, Relative_Depth, Functional_Correspondence.
  - Judge LLM: OpenAI-compatible API (e.g. lmdeploy api_server on port 23333).

Examples (from repo root):
  python JUDGE/eval_suite.py vip --input ../RESULT/.../results_....json --output vip_metrics.json
  python JUDGE/eval_suite.py blink --input ../RESULT/.../results_....json
  python JUDGE/eval_suite.py cvbench --input .../results_....json
  python JUDGE/eval_suite.py inst_it --mode image_mc --input ../RESULT/.../results_....json
  python JUDGE/eval_suite.py inst_it --mode image_oe --input ../RESULT/.../results_....json --output judged.json
  python JUDGE/eval_suite.py gar --kind mc --input ../RESULT/.../results_....json
  python JUDGE/eval_suite.py gar --kind simple --input ... --output gar_simple_metrics.json

InternVL-78B / VLMEvalKit (.xlsx): convert first —
  python JUDGE/xlsx_to_steering_json.py --input RESULT_VLMEvalKit/foo.xlsx --output foo.json
  (see JUDGE/xlsx_to_steering_json.py --help; use --dry-run to inspect columns)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Repo imports (run from repo root: python JUDGE/eval_suite.py ...)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.join(_REPO_ROOT, "JUDGE") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "JUDGE"))

from behaviors import VIP_BENCH_META_JSON, VIP_BENCH_QUESTIONS_JSONL
from evaluate_gar import (
    compute_metric_gar_detailed,
    compute_metric_gar_simple,
    compute_metric_mc_gar_vqa,
)
from evaluate_inst_it import (
    VLLMJudge,
    evaluate_mc_file,
    load_inst_it_image_mc_ground_truth,
    load_inst_it_image_oe_ground_truth,
    load_inst_it_video_mc_ground_truth,
    load_inst_it_video_oe_ground_truth,
    process_oe_json_file,
)
from eval_blink import compute_metrics, extract_answer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIP_JUDGE_PROMPT = """Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. <AND> in the ground truth means it is totally right only when all elements in the ground truth are present in the prediction, and <OR> means it is totally right when any one element in the ground truth is present in the prediction. The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Just complete the last space of the correctness score.

Question | Ground truth | Prediction | Correctness
--- | --- | --- | ---
What is x in the equation within the yellow rectangle? | -1 <AND> -5 | x = 3 | 0.0
What is x in the equation within the yellow rectangle? | -1 <AND> -5 | x = -1 | 0.5
What is x in the equation within the yellow rectangle? | -1 <AND> -5 | x = -5 | 0.5
What is x in the equation within the red rectangle? | -1 <AND> -5 | x = -5 or 5 | 0.5
What is x in the equation within the orange rectangle? | -1 <AND> -5 | x = -1 or x = -5 | 1.0
Can you explain this meme within the blue rectangle? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme talks about Iceland and Greenland. It's pointing out that despite their names, Iceland is not very icy and Greenland isn't very green. | 0.4
Can you explain this meme within the blue rectangle? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme is using humor to point out the misleading nature of Iceland's and Greenland's names. Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow. The text 'This is why I have trust issues' is a playful way to suggest that these contradictions can lead to distrust or confusion. The humor in this meme is derived from the unexpected contrast between the names of the countries and their actual physical characteristics. | 1.0
"""

# Default BLINK subsets (evalscope subset names / stored in result["type"])
DEFAULT_BLINK_SUBSETS = frozenset(
    {
        "Relative_Reflectance",
        "Relative_Depth",
        "Functional_Correspondence",
    }
)

_BLINK_ALIASES = {
    "relative_reflectance": "Relative_Reflectance",
    "relative_depth": "Relative_Depth",
    "reletive_depth": "Relative_Depth",
    "functional_correspondence": "Functional_Correspondence",
    "rrf": "Relative_Reflectance",
    "rdp": "Relative_Depth",
    "fc": "Functional_Correspondence",
}


def _norm_blink_type(t: str) -> str:
    t0 = (t or "").strip()
    if not t0:
        return ""
    if t0 in DEFAULT_BLINK_SUBSETS:
        return t0
    tl = re.sub(r"\s+", "_", t0)
    tl_lower = tl.lower()
    if tl_lower in _BLINK_ALIASES:
        return _BLINK_ALIASES[tl_lower]
    if tl in _BLINK_ALIASES:
        return _BLINK_ALIASES[tl]
    return t0


def _get_openai_client(base_url: str, api_key: str = "EMPTY"):
    from openai import OpenAI

    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)


def _load_json_list_or_dict(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _vip_build_questions_map(questions_jsonl: str) -> Dict[str, str]:
    questions: Dict[str, str] = {}
    with open(questions_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tmp = json.loads(line)
            qid = tmp["question_id"]
            questions[f"v1_{qid}"] = tmp["text"]
    return questions


def _vip_id_key_variants(sid: str) -> List[str]:
    """ViP meta keys may be '42' or 'v1_42'; results JSON may use either — register both."""
    s = str(sid).strip()
    if not s:
        return []
    keys = {s}
    if s.startswith("v1_"):
        keys.add(s[3:])
    else:
        keys.add(f"v1_{s}")
    return list(keys)


def _vip_row_primary_id(row: Dict[str, Any]) -> Optional[str]:
    for k in ("question_id", "sample_id", "image_id"):
        v = row.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


def _vip_prediction_map(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map metadata id -> prediction text (API models may only fill raw_model_output)."""
    out: Dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_id = _vip_row_primary_id(row)
        if raw_id is None:
            continue
        pred = (row.get("model_output") or row.get("raw_model_output") or "").strip()
        for k in _vip_id_key_variants(raw_id):
            out[k] = pred
    return out


def _vip_fallback_predictions_in_order(rows: List[Dict[str, Any]]) -> List[str]:
    """Fallback predictions for rows without explicit id, preserving JSON order."""
    out: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if _vip_row_primary_id(row) is not None:
            continue
        pred = (row.get("model_output") or row.get("raw_model_output") or "").strip()
        if pred:
            out.append(pred)
    return out


def _parse_score_token(content: str) -> float:
    token = content.split()[0].strip()
    return float(token)


def eval_vip_bbox_lmdeploy(
    results_path: str,
    out_path: Optional[str],
    base_url: str,
    model_name: str,
    meta_json: str,
    questions_jsonl: str,
    api_key: str = "EMPTY",
    num_run: int = 1,
) -> Dict[str, Any]:
    """
    ViP-Bench bbox split: LLM scores 0.0–1.0 per sample (same rubric as JUDGE/eval_vip.py).
    Expects `prompting_with_steering.py` JSON: a list of dicts with question_id + model_output.
    """
    raw = _load_json_list_or_dict(results_path)
    fallback_preds: List[str] = []
    if isinstance(raw, dict):
        # legacy dict keyed by id -> prediction string
        pred_map = {str(k): str(v) for k, v in raw.items()}
    else:
        pred_map = _vip_prediction_map(raw)
        fallback_preds = _vip_fallback_predictions_in_order(raw)
    fallback_idx = 0
    fallback_used = 0

    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    questions = _vip_build_questions_map(questions_jsonl)
    client = _get_openai_client(base_url, api_key=api_key)

    counter = Counter()
    len_data = 0
    for _, line in meta.items():
        cap = set(line.get("capability", []) or [])
        counter.update(cap)
        len_data += 1
    counter["total"] = len_data

    sorted_caps = [k for k, _ in counter.most_common() if k != "total"]

    grade_results: Dict[str, Dict[str, Any]] = {}

    for run_idx in range(num_run):
        for sample_id, line in tqdm(meta.items(), desc=f"ViP judge run {run_idx}"):
            if sample_id in pred_map:
                model_pred = pred_map[sample_id]
            elif fallback_idx < len(fallback_preds):
                model_pred = fallback_preds[fallback_idx]
                fallback_idx += 1
                fallback_used += 1
            else:
                print(f"[warn] missing prediction for id={sample_id}, score=0")
                score = 0.0
                content = ""
                model_pred = None

            if model_pred is not None:
                gt_ans = line["answer"].replace("<AND>", " <AND> ").replace("<OR>", " <OR> ")
                qtext = questions.get(sample_id, line.get("question", ""))
                user_line = VIP_JUDGE_PROMPT + "\n" + " | ".join([qtext, gt_ans, model_pred, ""])
                messages = [{"role": "user", "content": user_line}]
                temperature = 0.0
                try_time = 0
                score = 0.0
                content = ""
                while try_time < 6:
                    try:
                        resp = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            max_tokens=24,
                            temperature=temperature,
                        )
                        content = (resp.choices[0].message.content or "").strip()
                        try:
                            score = _parse_score_token(content)
                            if 0.0 <= score <= 1.0:
                                break
                            raise ValueError("out of range")
                        except Exception:
                            user_line = (
                                VIP_JUDGE_PROMPT
                                + "\n"
                                + " | ".join(
                                    [
                                        line.get("question", qtext),
                                        gt_ans,
                                        model_pred,
                                        "",
                                    ]
                                )
                                + "\nPredict the correctness of the answer (digit): "
                            )
                            messages = [{"role": "user", "content": user_line}]
                        try_time += 1
                        temperature += 0.5
                    except Exception as e:
                        print(f"[warn] id={sample_id} err={e}; sleep 5s")
                        time.sleep(5)
                        try_time += 1
                if try_time >= 6 and not (0.0 <= score <= 1.0):
                    score = 0.0

            rec = grade_results.setdefault(
                sample_id,
                {"model": [], "content": [], "score": []},
            )
            rec["model"].append(model_name)
            rec["content"].append(content)
            rec["score"].append(score)

    # Aggregate (mirror eval_vip.py)
    cap_keys = sorted_caps + ["total"]
    cap_scores = {k: [0.0] * num_run for k in cap_keys}
    for sid, v in grade_results.items():
        if sid not in meta:
            continue
        for i in range(num_run):
            sc = v["score"][i]
            for c in set(meta[sid].get("capability", []) or []):
                cap_scores[c][i] += sc
            cap_scores["total"][i] += sc

    cap_socres_pct: Dict[str, Any] = {}
    for k in cap_keys:
        denom = float(counter[k])
        cap_socres_pct[k] = np.array(cap_scores[k]) / denom * 100.0

    std = round(float(cap_socres_pct["total"].std()), 1) if len_data else 0.0
    row: Dict[str, Any] = {}
    for k in cap_keys:
        row[k] = round(float(np.mean(cap_socres_pct[k])), 1)
    row["std"] = std
    row["runs"] = str(list(np.round(cap_socres_pct["total"].copy(), 1)))

    df = pd.DataFrame([row])
    summary = {
        "task": "vip_image_oe_qa",
        "split": "bbox",
        "len_data": len_data,
        "fallback_sequential_used": fallback_used,
        "fallback_sequential_available": len(fallback_preds),
        "judge_base_url": base_url,
        "judge_model": model_name,
        "per_capability_percent": row,
        "dataframe": df.to_dict(orient="records"),
        "grade_results": grade_results,
    }

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        # grade_results can be large; write compact summary without full grade dump optional
        to_save = {k: v for k, v in summary.items() if k != "grade_results"}
        to_save["grade_results"] = grade_results
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
        print(f"[vip] wrote {out_path}")
    print(df)
    return summary


def _blink_cvbench_rows_for_metrics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """`prompting_with_steering.py` saves answer/model_output; `compute_metrics` expects ground_truth/extracted_prediction."""
    out: List[Dict[str, Any]] = []
    for item in rows:
        r = dict(item)
        gt = (item.get("ground_truth") or item.get("answer") or "").strip().upper()
        r["ground_truth"] = gt
        r["extracted_prediction"] = extract_answer((item.get("model_output") or "").strip())
        out.append(r)
    return out


def eval_blink_subset(
    results_path: str,
    subsets: Optional[Sequence[str]],
    out_metrics: Optional[str],
    *,
    mode: str = "blink_image_mc_qa",
    base_url: str = "http://127.0.0.1:23333/v1",
    model_name: str = "Qwen/Qwen2.5-72B-Instruct-AWQ",
    num_workers: int = 8,
) -> Dict[str, Any]:
    raw = _load_json_list_or_dict(results_path)
    if not isinstance(raw, list):
        raise ValueError("BLINK results must be a JSON list of rows.")
    allow = DEFAULT_BLINK_SUBSETS if not subsets else frozenset(subsets)
    filtered: List[Dict[str, Any]] = []
    for item in raw:
        t = _norm_blink_type(str(item.get("type", "") or item.get("sub_task", "") or item.get("l2-category", "")))
        if t in allow:
            filtered.append(item)
    if not filtered:
        seen = sorted({_norm_blink_type(str(x.get("type", "") or x.get("sub_task", ""))) for x in raw if isinstance(x, dict)})
        raise RuntimeError(
            f"No BLINK rows left after filtering to {allow}. "
            f"Found type/sub_task values (normalized sample): {seen[:40]}{'...' if len(seen) > 40 else ''}. "
            "Ensure rows include BLINK sub_task in \"type\" (or \"sub_task\")."
        )
    rows_for_m = _blink_cvbench_rows_for_metrics(filtered)
    metrics = compute_metrics(rows_for_m)
    metrics["blink_subsets_used"] = sorted(allow)
    metrics["num_rows_filtered"] = len(filtered)
    metrics["eval_mode"] = mode
    metrics["mc_extract"] = "regex"
    print(json.dumps(metrics["overall"], indent=2, ensure_ascii=False))
    if out_metrics:
        with open(out_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[blink] wrote {out_metrics}")
    return metrics


def eval_cvbench(
    results_path: str,
    out_metrics: Optional[str],
    *,
    mode: str = "cvbench_image_mc_qa",
    base_url: str = "http://127.0.0.1:23333/v1",
    model_name: str = "Qwen/Qwen2.5-72B-Instruct-AWQ",
    num_workers: int = 8,
) -> Dict[str, Any]:
    raw = _load_json_list_or_dict(results_path)
    if not isinstance(raw, list):
        raise ValueError("CV-Bench results must be a JSON list.")
    rows_for_m = _blink_cvbench_rows_for_metrics(raw)
    metrics = compute_metrics(rows_for_m)
    metrics["eval_mode"] = mode
    metrics["mc_extract"] = "regex"
    if out_metrics:
        with open(out_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[cvbench] wrote {out_metrics}")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Subset evaluation + lmdeploy judge (see module docstring).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_vip = sub.add_parser("vip", help="ViP-Bench bbox open-ended (LLM judge, 0–1 scores).")
    p_vip.add_argument("--input", required=True, help="results_*.json from prompting_with_steering.py")
    p_vip.add_argument("--output", default=None, help="optional JSON metrics + grades")
    p_vip.add_argument("--base_url", default="http://127.0.0.1:23333/v1", help="OpenAI-compatible base URL")
    p_vip.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-72B-Instruct-AWQ",
        help="Served model name (lmdeploy)",
    )
    p_vip.add_argument("--meta_json", default=VIP_BENCH_META_JSON)
    p_vip.add_argument("--questions_jsonl", default=VIP_BENCH_QUESTIONS_JSONL)
    p_vip.add_argument("--api_key", default="EMPTY")
    p_vip.add_argument("--num_run", type=int, default=1)

    p_blink = sub.add_parser("blink", help="BLINK MC: only Relative_Reflectance / Relative_Depth / Functional_Correspondence.")
    p_blink.add_argument("--input", required=True)
    p_blink.add_argument("--output", default=None, help="optional metrics JSON")
    p_blink.add_argument(
        "--mode",
        default="blink_image_mc_qa",
        choices=["blink_image_mc_qa"],
        help="BLINK MC task id (for metrics metadata only)",
    )
    p_blink.add_argument(
        "--subsets",
        nargs="*",
        default=None,
        help="override subset names (default: three Relative_* + Functional_Correspondence)",
    )
    p_blink.add_argument("--base_url", default="http://127.0.0.1:23333/v1")
    p_blink.add_argument("--model_name", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    p_blink.add_argument("--num_workers", type=int, default=8)

    p_cv = sub.add_parser("cvbench", help="CV-Bench MC (full file).")
    p_cv.add_argument("--input", required=True)
    p_cv.add_argument("--output", default=None)
    p_cv.add_argument(
        "--mode",
        default="cvbench_image_mc_qa",
        choices=["cvbench_image_mc_qa"],
        help="CV-Bench MC task id (for metrics metadata only)",
    )
    p_cv.add_argument("--base_url", default="http://127.0.0.1:23333/v1")
    p_cv.add_argument("--model_name", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    p_cv.add_argument("--num_workers", type=int, default=8)

    p_inst = sub.add_parser("inst_it", help="Inst-It-Bench: MC (no LLM) or OE (lmdeploy judge).")
    p_inst.add_argument(
        "--mode",
        required=True,
        choices=[
            "image_mc",
            "video_mc",
            "image_oe",
            "video_oe",
        ],
    )
    p_inst.add_argument("--input", required=True, help="one results JSON file")
    p_inst.add_argument(
        "--output",
        default=None,
        help="MC: scored JSON; OE: judged JSON path (default: input with _evaluated.json)",
    )
    p_inst.add_argument("--base_url", default="http://127.0.0.1:23333/v1")
    p_inst.add_argument("--model_name", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    p_inst.add_argument("--num_workers", type=int, default=8)
    p_inst.add_argument("--overwrite", action="store_true")

    p_gar = sub.add_parser("gar", help="GAR: MC (no LLM) or simple/detail (lmdeploy True/False).")
    p_gar.add_argument("--kind", required=True, choices=["mc", "simple", "detailed"])
    p_gar.add_argument("--input", required=True)
    p_gar.add_argument("--output", default=None, help="metrics JSON path")
    p_gar.add_argument("--base_url", default="http://127.0.0.1:23333/v1")
    p_gar.add_argument("--model_name", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    p_gar.add_argument("--num_process", type=int, default=8)

    args = p.parse_args()

    if args.cmd == "vip":
        eval_vip_bbox_lmdeploy(
            args.input,
            args.output,
            args.base_url,
            args.model_name,
            args.meta_json,
            args.questions_jsonl,
            api_key=args.api_key,
            num_run=args.num_run,
        )
    elif args.cmd == "blink":
        eval_blink_subset(
            args.input,
            args.subsets,
            args.output,
            mode=args.mode,
            base_url=args.base_url,
            model_name=args.model_name,
            num_workers=args.num_workers,
        )
    elif args.cmd == "cvbench":
        eval_cvbench(
            args.input,
            args.output,
            mode=args.mode,
            base_url=args.base_url,
            model_name=args.model_name,
            num_workers=args.num_workers,
        )
    elif args.cmd == "inst_it":
        if args.mode.endswith("_mc"):
            if args.mode == "image_mc":
                gt_list = load_inst_it_image_mc_ground_truth()
                split = "image"
            else:
                gt_list = load_inst_it_video_mc_ground_truth()
                split = "video"
            out = args.output or (args.input.replace(".json", "_mc_eval.json"))
            evaluate_mc_file(
                args.input,
                out,
                gt_list,
                split=split,
                match_by_question_id=True,
            )
            print(f"[inst_it {args.mode}] wrote {out}")
        else:
            gt_loaders = {
                "image_oe": load_inst_it_image_oe_ground_truth,
                "video_oe": load_inst_it_video_oe_ground_truth,
            }
            gt_list = gt_loaders[args.mode]()
            split = "image" if args.mode == "image_oe" else "video"
            out = args.output or (args.input.replace(".json", "_evaluated.json"))
            judge = VLLMJudge(base_url=args.base_url, model_name=args.model_name)
            process_oe_json_file(
                args.input,
                out,
                judge,
                num_workers=args.num_workers,
                overwrite=args.overwrite,
                gt_list=gt_list,
                split=split,
                match_by_question_id=True,
            )
            print(f"[inst_it {args.mode}] wrote {out}")
    elif args.cmd == "gar":
        with open(args.input, "r", encoding="utf-8") as f:
            preds = json.load(f)
        out = args.output or (args.input.replace(".json", f"_{args.kind}_metrics.json"))
        if args.kind == "mc":
            compute_metric_mc_gar_vqa(preds, out)
        elif args.kind == "simple":
            compute_metric_gar_simple(
                preds,
                out,
                base_url=args.base_url,
                model_name=args.model_name,
                num_process=args.num_process,
            )
        else:
            compute_metric_gar_detailed(
                preds,
                out,
                base_url=args.base_url,
                model_name=args.model_name,
                num_process=args.num_process,
            )
        print(f"[gar {args.kind}] wrote {out}")


if __name__ == "__main__":
    main()
