"""
Batch-evaluate all steering / API / gar_qwen3vl / VLMEvalKit xlsx results under RESULT/.

- Traversal order (phase): noref → gar_qwen3vl → gemini → o3 →
  lam0.25 → lam0.5 → lam2.0 → cross_model_steer → pixelrefer_mdvp →
  pixelrefer_osprey → no_marker → (everything else last).

- Every JSON is evaluated and metrics saved; files that share the same
  (task type, multiplier, model_name, model_size, parent experiment folder) but
  differ only by layer get a group summary with the best layer and max metric.

Usage (from repo root):
  python JUDGE/batch_eval_results.py --result-root ../RESULT --out-dir ../RESULT/_eval_metrics
  python JUDGE/batch_eval_results.py --result-root ../RESULT --dry-run
  python JUDGE/batch_eval_results.py --phase noref gemini          # only those phases (omit --phase = all files)
  python JUDGE/batch_eval_results.py --smoke --smoke-limit 2        # temp dir, no writes under out-dir

Requires lmdeploy (or compatible OpenAI API) for judges, same as eval_suite.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_JUDGE = Path(__file__).resolve().parent
if str(_JUDGE) not in sys.path:
    sys.path.insert(0, str(_JUDGE))

from tqdm import tqdm

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
from eval_suite import (
    eval_blink_subset,
    eval_cvbench,
    eval_vip_bbox_lmdeploy,
)


# ---------------------------------------------------------------------------
# Phase ordering (lower = earlier). First matching rule wins.
# ---------------------------------------------------------------------------

PHASE_CHOICES = (
    "noref",
    "gar_qwen3vl",
    "gemini",
    "o3",
    "lam0.25",
    "lam0.5",
    "lam2.0",
    "cross_model",
    "pixelrefer_mdvp",
    "pixelrefer_osprey",
    "no_marker",
)

ALL_TASK_LABELS = (
    "blink_image_mc_qa",
    "cvbench_image_mc_qa",
    "vip_image_oe_qa",
    "inst_it_image_mc_qa",
    "inst_it_video_mc_qa",
    "inst_it_image_oe_qa",
    "inst_it_video_oe_qa",
    "gar_image_mc_qa",
    "gar_image_simple_oe_qa",
    "gar_image_detail_oe_qa",
)

TASK_TOKEN_ALIASES: Dict[str, Set[str]] = {
    "blink": {"blink_image_mc_qa"},
    "cvbench": {"cvbench_image_mc_qa"},
    "vip": {"vip_image_oe_qa"},
    "inst": {
        "inst_it_image_mc_qa",
        "inst_it_video_mc_qa",
        "inst_it_image_oe_qa",
        "inst_it_video_oe_qa",
    },
    "inst_mc": {"inst_it_image_mc_qa", "inst_it_video_mc_qa"},
    "inst_oe": {"inst_it_image_oe_qa", "inst_it_video_oe_qa"},
    "gar": {"gar_image_mc_qa", "gar_image_simple_oe_qa", "gar_image_detail_oe_qa"},
    "gar_mc": {"gar_image_mc_qa"},
    "gar_oe": {"gar_image_simple_oe_qa", "gar_image_detail_oe_qa"},
}


def phase_info(path: str) -> Tuple[int, Optional[str]]:
    """Same rule order as historical phase_rank; returns (rank, phase name or None if no rule matched)."""
    p = path.replace("\\", "/").lower()
    if "gar_qwen3vl" in p or "gar_vcd_qwen3vl" in p:
        return 1, "gar_qwen3vl"
    if "gemini-2.5-pro_api" in p or "gemini-2.5-pro" in p:
        return 2, "gemini"
    if "o3_api" in p or re.search(r"/o3/", p):
        return 3, "o3"
    if "lam0.25" in p:
        return 4, "lam0.25"
    if "lam0.5" in p:
        return 5, "lam0.5"
    if "lam2.0" in p:
        return 6, "lam2.0"
    if "cross_model_steer" in p:
        return 7, "cross_model"
    if "pixelrefer_mdvp" in p:
        return 8, "pixelrefer_mdvp"
    if "pixelrefer_osprey" in p:
        return 9, "pixelrefer_osprey"
    if "no_marker" in p:
        return 10, "no_marker"
    if "noref" in p and "no_marker" not in p:
        return 0, "noref"
    return 11, None


def phase_rank(path: str) -> int:
    return phase_info(path)[0]


def parse_phase_filter(phase_args: Optional[Sequence[str]]) -> Optional[Set[str]]:
    """None or empty → no filter (all files). Legacy: --phase all alone → no filter."""
    if not phase_args:
        return None
    s = {x.strip().lower() for x in phase_args if x and str(x).strip()}
    if not s:
        return None
    if "all" in s:
        if len(s) > 1:
            raise SystemExit("Cannot combine 'all' with other --phase names; omit --phase to run everything.")
        return None
    unknown = s - set(PHASE_CHOICES)
    if unknown:
        raise SystemExit(
            f"Unknown --phase value(s): {sorted(unknown)}. Valid: {', '.join(PHASE_CHOICES)} (omit --phase for all files)"
        )
    return s


def path_matches_phases(path_str: str, selected: Optional[Set[str]]) -> bool:
    if selected is None:
        return True
    _, name = phase_info(path_str)
    if name is None:
        return False
    return name in selected


def phase_tag_for_filename(selected: Optional[Set[str]]) -> str:
    """Build a stable file-name tag from --phase selection."""
    if not selected:
        return "all"
    ordered = [p for p in PHASE_CHOICES if p in selected]
    if not ordered:
        return "custom"
    # Keep dots in lam0.25 for readability; no spaces/slashes in phase names.
    return "+".join(ordered)


def _task_label_for_xlsx(path: Path) -> str:
    s = path.stem.upper()
    if "BLINK" in s:
        return "blink_image_mc_qa"
    if "VIP" in s or "VIPBENCH" in s.replace(" ", ""):
        return "vip_image_oe_qa"
    return "xlsx"


def _expand_task_token(token: str) -> Set[str]:
    t = token.strip().lower()
    if not t:
        return set()
    if t in ("all", "*"):
        return set(ALL_TASK_LABELS)
    if t in TASK_TOKEN_ALIASES:
        return set(TASK_TOKEN_ALIASES[t])
    if t in ALL_TASK_LABELS:
        return {t}
    raise SystemExit(
        f"Unknown task token in --phase-task: {token!r}. "
        f"Known aliases: {sorted(TASK_TOKEN_ALIASES.keys())}; direct task labels: {list(ALL_TASK_LABELS)}"
    )


def parse_phase_task_rules(specs: Optional[Sequence[str]]) -> Dict[str, Set[str]]:
    """
    Parse --phase-task entries like:
      gemini=blink,vip
      o3=inst,gar
    """
    out: Dict[str, Set[str]] = {}
    if not specs:
        return out
    for raw in specs:
        if "=" not in raw:
            raise SystemExit(f"Invalid --phase-task {raw!r}; expected PHASE=task1,task2")
        phase_part, task_part = raw.split("=", 1)
        phase = phase_part.strip().lower()
        if phase not in PHASE_CHOICES:
            raise SystemExit(f"Invalid --phase-task phase {phase!r}; choose from {PHASE_CHOICES}")
        tokens = [x.strip() for x in task_part.split(",") if x.strip()]
        if not tokens:
            raise SystemExit(f"Invalid --phase-task {raw!r}; no tasks listed")
        allowed = out.setdefault(phase, set())
        for tk in tokens:
            allowed.update(_expand_task_token(tk))
    return out


def path_matches_phase_task_rules(
    path: Path,
    phase_task_rules: Dict[str, Set[str]],
) -> bool:
    if not phase_task_rules:
        return True
    _, phase_name = phase_info(str(path))
    if phase_name is None:
        return True
    allowed = phase_task_rules.get(phase_name)
    if not allowed:
        return True
    if path.suffix.lower() == ".xlsx":
        label = _task_label_for_xlsx(path)
    else:
        label = parse_type_from_results_filename(path.name) or ""
    return label in allowed


# ---------------------------------------------------------------------------
# Smoke: truncate JSON / xlsx rows; no persistent out-dir
# ---------------------------------------------------------------------------

def truncate_json_payload(data: Any, limit: int) -> Any:
    if limit <= 0:
        return data
    if isinstance(data, list):
        return data[:limit]
    if isinstance(data, dict):
        out = dict(data)
        for k in ("results", "data", "predictions", "preds"):
            if k in out and isinstance(out[k], list):
                out[k] = out[k][:limit]
                return out
        keys = list(out.keys())[:limit]
        return {k: out[k] for k in keys}
    return data


def write_smoke_json(src: Path, dst: Path, limit: int) -> None:
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = truncate_json_payload(data, limit)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Filename → task type
# ---------------------------------------------------------------------------

_MC_EVAL_TASK_ALIASES = {
    "gar_image_think_mc_qa": "gar_image_mc_qa",
    "blink_image_think_mc_qa": "blink_image_mc_qa",
    "cvbench_image_think_mc_qa": "cvbench_image_mc_qa",
    "inst_it_image_think_mc_qa": "inst_it_image_mc_qa",
    "inst_it_video_think_mc_qa": "inst_it_video_mc_qa",
}


def normalize_eval_task(task: Optional[str]) -> Optional[str]:
    """Map legacy *think_mc* result types to standard MC eval tasks."""
    if task is None:
        return None
    return _MC_EVAL_TASK_ALIASES.get(task, task)


def parse_type_from_results_filename(name: str) -> Optional[str]:
    if not name.endswith(".json"):
        return None
    if name.startswith("results_gar_"):
        if "_mc_" in name or name.endswith("_mc.json") or "gar_mc" in name:
            return "gar_image_mc_qa"
        if "simple" in name:
            return "gar_image_simple_oe_qa"
        if "detailed" in name:
            return "gar_image_detail_oe_qa"
        return None
    if "_override_vector_" in name:
        m = re.search(r"type_(.+?)_override_vector", name)
        return m.group(1) if m else None
    m = re.search(r"type_(.+?)_model_name_", name)
    return m.group(1) if m else None


def parse_layer_mult(name: str) -> Tuple[Optional[int], Optional[str]]:
    m = re.search(r"results_layer_(\d+)_multiplier_([0-9.]+)_", name)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def group_key_for_layer_file(path: Path) -> Optional[str]:
    """Same parent dir + same multiplier/task/model tail, ignoring layer index."""
    name = path.name
    m = re.match(
        r"^results_layer_\d+(_multiplier_[0-9.]+_behavior_refer_type_.+)\.json$",
        name,
    )
    if not m:
        return None
    return f"{path.parent.as_posix()}|{m.group(1)}"


# ---------------------------------------------------------------------------
# Metric extraction (single scalar for max-over-layers)
# ---------------------------------------------------------------------------

def primary_metric_from_eval(task: str, payload: Any) -> Optional[float]:
    if payload is None:
        return None
    if isinstance(payload, dict):
        if "accuracy" in payload and isinstance(payload["accuracy"], (int, float)):
            return float(payload["accuracy"])
        if "overall" in payload and isinstance(payload["overall"], dict):
            oa = payload["overall"].get("accuracy")
            if isinstance(oa, (int, float)):
                return float(oa)
        if "per_capability_percent" in payload:
            row = payload["per_capability_percent"]
            if isinstance(row, dict) and "total" in row:
                return float(row["total"])
        if isinstance(payload.get("accuracy"), str):
            return None
    return None


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------

def _out_path_for(in_path: Path, out_root: Path, result_root: Path) -> Path:
    rel = in_path.relative_to(result_root)
    return out_root / rel.parent / (rel.name + ".metrics.json")


def eval_one_json(
    json_path: Path,
    out_path: Path,
    task: str,
    *,
    base_url: str,
    judge_model: str,
    num_workers: int,
    skip_vip: bool,
) -> Tuple[Any, Optional[float]]:
    """Returns (raw_result_or_metrics, primary_float)."""
    os.makedirs(out_path.parent, exist_ok=True)
    task = normalize_eval_task(task) or task

    if task == "vip_image_oe_qa":
        if skip_vip:
            return {"skipped": True}, None
        summ = eval_vip_bbox_lmdeploy(
            str(json_path),
            str(out_path),
            base_url,
            judge_model,
            VIP_BENCH_META_JSON,
            VIP_BENCH_QUESTIONS_JSONL,
        )
        row = summ.get("per_capability_percent") or {}
        pm = float(row.get("total", 0)) if isinstance(row, dict) else None
        return summ, pm

    if task == "blink_image_mc_qa":
        m = eval_blink_subset(
            str(json_path),
            None,
            str(out_path),
            mode="blink_image_mc_qa",
            base_url=base_url,
            model_name=judge_model,
            num_workers=num_workers,
        )
        return m, primary_metric_from_eval(task, m)

    if task == "cvbench_image_mc_qa":
        m = eval_cvbench(
            str(json_path),
            str(out_path),
            mode="cvbench_image_mc_qa",
            base_url=base_url,
            model_name=judge_model,
            num_workers=num_workers,
        )
        return m, primary_metric_from_eval(task, m)

    if task == "inst_it_image_mc_qa":
        gt = load_inst_it_image_mc_ground_truth()
        tmp_out = str(out_path).replace(".metrics.json", "_mc_eval.json")
        r = evaluate_mc_file(
            str(json_path),
            tmp_out,
            gt,
            split="image",
            match_by_question_id=True,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"mc_eval": r, "tmp_json": tmp_out}, f, ensure_ascii=False, indent=2)
        return r, float(r.get("accuracy", 0))

    if task == "inst_it_video_mc_qa":
        gt = load_inst_it_video_mc_ground_truth()
        tmp_out = str(out_path).replace(".metrics.json", "_mc_eval.json")
        r = evaluate_mc_file(
            str(json_path),
            tmp_out,
            gt,
            split="video",
            match_by_question_id=True,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"mc_eval": r, "tmp_json": tmp_out}, f, ensure_ascii=False, indent=2)
        return r, float(r.get("accuracy", 0))

    if task == "inst_it_image_oe_qa":
        gt = load_inst_it_image_oe_ground_truth()
        judge = VLLMJudge(base_url=base_url, model_name=judge_model)
        tmp_out = str(out_path).replace(".metrics.json", "_evaluated.json")
        process_oe_json_file(
            str(json_path),
            tmp_out,
            judge,
            num_workers=num_workers,
            overwrite=True,
            gt_list=gt,
            split="image",
            match_by_question_id=True,
        )
        with open(tmp_out, "r", encoding="utf-8") as f:
            data = json.load(f)
        scores = [float(x.get("score", 0)) for x in data if isinstance(x, dict) and "score" in x]
        avg = sum(scores) / len(scores) if scores else 0.0
        summary = {"avg_score": avg, "n": len(scores), "evaluated_path": tmp_out}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary, avg * 100.0

    if task == "inst_it_video_oe_qa":
        gt = load_inst_it_video_oe_ground_truth()
        judge = VLLMJudge(base_url=base_url, model_name=judge_model)
        tmp_out = str(out_path).replace(".metrics.json", "_evaluated.json")
        process_oe_json_file(
            str(json_path),
            tmp_out,
            judge,
            num_workers=num_workers,
            overwrite=True,
            gt_list=gt,
            split="video",
            match_by_question_id=True,
        )
        with open(tmp_out, "r", encoding="utf-8") as f:
            data = json.load(f)
        scores = [float(x.get("score", 0)) for x in data if isinstance(x, dict) and "score" in x]
        avg = sum(scores) / len(scores) if scores else 0.0
        summary = {"avg_score": avg, "n": len(scores), "evaluated_path": tmp_out}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary, avg * 100.0

    if task == "gar_image_mc_qa":
        with open(json_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        res = compute_metric_mc_gar_vqa(preds, str(out_path))
        acc = res.get("accuracy")
        pm = float(acc) if isinstance(acc, (int, float)) else None
        return res, pm

    if task == "gar_image_simple_oe_qa":
        with open(json_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        res = compute_metric_gar_simple(
            preds,
            str(out_path),
            base_url=base_url,
            model_name=judge_model,
            num_process=num_workers,
        )
        acc = res.get("accuracy")
        pm = float(acc) if isinstance(acc, (int, float)) else None
        return res, pm

    if task == "gar_image_detail_oe_qa":
        with open(json_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        res = compute_metric_gar_detailed(
            preds,
            str(out_path),
            base_url=base_url,
            model_name=judge_model,
            num_process=num_workers,
        )
        acc = res.get("accuracy")
        pm = float(acc) if isinstance(acc, (int, float)) else None
        return res, pm

    return {"error": f"unsupported_task:{task}"}, None


def xlsx_to_eval(
    xlsx_path: Path,
    out_metrics: Path,
    tmp_root: Path,
    base_url: str,
    judge_model: str,
    num_workers: int,
    row_limit: Optional[int] = None,
) -> Tuple[Any, Optional[float]]:
    from xlsx_to_steering_json import xlsx_to_rows

    stem = xlsx_path.stem.upper()
    tmp_json = tmp_root / (xlsx_path.stem + ".json")
    tmp_json.parent.mkdir(parents=True, exist_ok=True)
    rows = xlsx_to_rows(str(xlsx_path), sheet=None, col_overrides=None)
    if row_limit is not None and row_limit > 0:
        rows = rows[:row_limit]
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    os.makedirs(out_metrics.parent, exist_ok=True)

    if "BLINK" in stem:
        m = eval_blink_subset(
            str(tmp_json),
            None,
            str(out_metrics),
            mode="blink_image_mc_qa",
            base_url=base_url,
            model_name=judge_model,
            num_workers=num_workers,
        )
        return m, primary_metric_from_eval("blink", m)
    if "VIP" in stem or "VIPBENCH" in stem.replace(" ", ""):
        summ = eval_vip_bbox_lmdeploy(
            str(tmp_json),
            str(out_metrics),
            base_url,
            judge_model,
            VIP_BENCH_META_JSON,
            VIP_BENCH_QUESTIONS_JSONL,
        )
        row = summ.get("per_capability_percent", {})
        pm = float(row.get("total", 0)) if isinstance(row, dict) else None
        return summ, pm

    if "INST" in stem and "VIDEO" in stem:
        gt = load_inst_it_video_mc_ground_truth()
        mc_out = str(out_metrics).replace(".metrics.json", "_mc_eval.json")
        r = evaluate_mc_file(
            str(tmp_json),
            mc_out,
            gt,
            split="video",
            match_by_question_id=True,
        )
        with open(out_metrics, "w", encoding="utf-8") as f:
            json.dump({"mc_eval": r}, f, indent=2, ensure_ascii=False)
        return r, float(r.get("accuracy", 0))
    if "INST" in stem and "IMAGE" in stem:
        gt = load_inst_it_image_mc_ground_truth()
        mc_out = str(out_metrics).replace(".metrics.json", "_mc_eval.json")
        r = evaluate_mc_file(
            str(tmp_json),
            mc_out,
            gt,
            split="image",
            match_by_question_id=True,
        )
        with open(out_metrics, "w", encoding="utf-8") as f:
            json.dump({"mc_eval": r}, f, indent=2, ensure_ascii=False)
        return r, float(r.get("accuracy", 0))

    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump({"error": "unmapped_xlsx", "stem": xlsx_path.stem}, f)
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch eval all RESULT json/xlsx (ordered phases, layer max).")
    parser.add_argument(
        "--result-root",
        type=str,
        default=str(_REPO.parent / "RESULT"),
        help="Path to RESULT directory (default: ../RESULT from repo)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Mirror tree for metrics (default: <result-root>/_eval_metrics; ignored with --smoke)",
    )
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:23333/v1")
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--skip-vip", action="store_true", help="Skip ViP (slow LLM judge).")
    parser.add_argument(
        "--skip-existing-metrics",
        action="store_true",
        default=True,
        help="Skip evaluation if target *.metrics.json already exists (default: True).",
    )
    parser.add_argument(
        "--overwrite-existing-metrics",
        action="store_true",
        help="Force re-evaluation even when *.metrics.json exists.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List files and phases only.")
    parser.add_argument(
        "--phase",
        nargs="*",
        metavar="PHASE",
        default=None,
        help=(
            "Restrict to one or more phases (repeatable list). "
            f"Names: {', '.join(PHASE_CHOICES)}. Omit --phase to run every file (incl. unmatched paths)."
        ),
    )
    parser.add_argument(
        "--phase-task",
        action="append",
        default=None,
        help=(
            "Per-phase task filter. Repeat this option. Format: PHASE=task1,task2. "
            "Aliases: blink/cvbench/vip/inst/inst_mc/inst_oe/gar/gar_mc/gar_oe; "
            "or direct task labels like inst_it_image_mc_qa."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a tiny slice per file (see --smoke-limit) under a temp directory; "
            "does not write metrics or summary under --out-dir. Prints a short JSON summary to stdout."
        ),
    )
    parser.add_argument("--smoke-limit", type=int, default=2, help="Max rows per JSON / xlsx in --smoke (default: 2).")
    parser.add_argument(
        "--smoke-workers",
        type=int,
        default=2,
        help="Worker count during --smoke (default: 2).",
    )
    parser.add_argument(
        "--smoke-eval-vip",
        action="store_true",
        help="With --smoke: run full ViP judge (slow); default is to skip ViP in smoke.",
    )
    args = parser.parse_args()
    if args.overwrite_existing_metrics:
        args.skip_existing_metrics = False

    result_root = Path(args.result_root).resolve()
    phase_filter = parse_phase_filter(args.phase)
    phase_task_rules = parse_phase_task_rules(args.phase_task)
    out_root_default = result_root / "_eval_metrics"
    out_root = Path(args.out_dir).resolve() if args.out_dir else out_root_default

    json_files = []
    for p in result_root.rglob("*.json"):
        n = p.name
        if n.endswith(".metrics.json") or "_mc_eval.json" in n or n.endswith("_evaluated.json"):
            continue
        if n.startswith("results_") or n.startswith("results_gar_"):
            json_files.append(p)
    xlsx_files = [p for p in result_root.rglob("*.xlsx") if p.is_file()]

    all_files: List[Path] = sorted(
        set(json_files) | set(xlsx_files),
        key=lambda x: (phase_rank(str(x)), str(x)),
    )
    if phase_filter is not None:
        all_files = [p for p in all_files if path_matches_phases(str(p), phase_filter)]
    if phase_task_rules:
        all_files = [p for p in all_files if path_matches_phase_task_rules(p, phase_task_rules)]

    if args.dry_run:
        for p in all_files:
            r, name = phase_info(str(p))
            disp = name if name is not None else "-"
            print(f"[phase {r:2d} {disp:18s}] {p.relative_to(result_root)}")
        print(f"Total: {len(all_files)} files under {result_root}")
        return

    workers = args.smoke_workers if args.smoke else args.num_workers

    def run_batch(out_root: Path, tmp_smoke: Optional[Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        records_local: List[Dict[str, Any]] = []
        layer_groups_local: Dict[str, List[Dict[str, Any]]] = {}
        skip_vip_effective = args.skip_vip or (args.smoke and not args.smoke_eval_vip)

        for p in tqdm(all_files, desc="batch_eval", unit="file"):
            rel = str(p.relative_to(result_root))
            try:
                if p.suffix.lower() == ".xlsx":
                    out_p = out_root / p.relative_to(result_root).with_suffix(".metrics.json")
                    if args.skip_existing_metrics and out_p.exists():
                        records_local.append(
                            {
                                "path": rel,
                                "kind": "xlsx",
                                "out": str(out_p),
                                "smoke": bool(args.smoke),
                                "skipped": "existing_metrics",
                            }
                        )
                        continue
                    tmp_root = (tmp_smoke or out_root) / "_tmp_xlsx"
                    row_lim = args.smoke_limit if args.smoke else None
                    payload, pm = xlsx_to_eval(
                        p,
                        out_p,
                        tmp_root,
                        args.base_url,
                        args.judge_model,
                        workers,
                        row_limit=row_lim,
                    )
                    records_local.append(
                        {
                            "path": rel,
                            "kind": "xlsx",
                            "primary_metric": pm,
                            "out": str(out_p),
                            "smoke": bool(args.smoke),
                        }
                    )
                    continue

                task = parse_type_from_results_filename(p.name)
                if not task:
                    records_local.append({"path": rel, "error": "unknown_task_filename"})
                    continue

                if args.smoke and task == "vip_image_oe_qa" and not args.smoke_eval_vip:
                    records_local.append(
                        {
                            "path": rel,
                            "task": task,
                            "smoke": True,
                            "skipped": "vip_smoke_default",
                            "hint": "ViP always iterates full meta; use --smoke-eval-vip to run full judge (slow).",
                        }
                    )
                    continue

                json_in = p
                if args.smoke and tmp_smoke is not None:
                    if task == "vip_image_oe_qa" and args.smoke_eval_vip:
                        json_in = p
                    else:
                        safe = rel.replace("\\", "/").replace("/", "__")
                        json_in = tmp_smoke / f"in_{safe}"
                        write_smoke_json(p, json_in, args.smoke_limit)

                out_p = out_root / p.relative_to(result_root).with_suffix(".metrics.json")
                if args.skip_existing_metrics and out_p.exists():
                    records_local.append(
                        {
                            "path": rel,
                            "task": task,
                            "out": str(out_p),
                            "smoke": bool(args.smoke),
                            "skipped": "existing_metrics",
                        }
                    )
                    continue
                os.makedirs(out_p.parent, exist_ok=True)
                payload, pm = eval_one_json(
                    json_in,
                    out_p,
                    task,
                    base_url=args.base_url,
                    judge_model=args.judge_model,
                    num_workers=workers,
                    skip_vip=skip_vip_effective,
                )
                rec: Dict[str, Any] = {
                    "path": rel,
                    "task": task,
                    "primary_metric": pm,
                    "out": str(out_p),
                    "smoke": bool(args.smoke),
                }
                layer, mult = parse_layer_mult(p.name)
                gk = group_key_for_layer_file(p)
                if gk and pm is not None:
                    rec["layer"] = layer
                    rec["multiplier"] = mult
                    layer_groups_local.setdefault(gk, []).append(
                        {
                            "path": rel,
                            "layer": layer,
                            "multiplier": mult,
                            "primary_metric": pm,
                            "metrics_file": str(out_p),
                        }
                    )
                records_local.append(rec)
            except Exception as e:
                records_local.append({"path": rel, "error": f"{type(e).__name__}: {e}", "smoke": bool(args.smoke)})

        best_layer: List[Dict[str, Any]] = []
        for gk, items in layer_groups_local.items():
            if len(items) < 2:
                continue
            best = max(items, key=lambda x: x["primary_metric"] if x["primary_metric"] is not None else -1)
            best_layer.append({"group": gk, "best": best, "all": items})
        return records_local, best_layer

    if args.smoke:
        with tempfile.TemporaryDirectory(prefix="batch_eval_smoke_") as td:
            td_path = Path(td)
            out_root = td_path / "out"
            out_root.mkdir(parents=True, exist_ok=True)
            records, best_layer = run_batch(out_root, tmp_smoke=td_path)
        summary = {
            "result_root": str(result_root),
            "smoke": True,
            "smoke_limit": args.smoke_limit,
            "phase_filter": sorted(phase_filter) if phase_filter else None,
            "phase_task_rules": {k: sorted(v) for k, v in phase_task_rules.items()},
            "n_files": len(all_files),
            "records": records,
            "layer_group_max": best_layer,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        print("[smoke] No files written under RESULT; temp workspace deleted.", file=sys.stderr)
        return

    out_root.mkdir(parents=True, exist_ok=True)
    records, best_layer = run_batch(out_root, tmp_smoke=None)

    summary_phase_tag = phase_tag_for_filename(phase_filter)
    summary_path = out_root / f"batch_eval_summary.phase_{summary_phase_tag}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "result_root": str(result_root),
                "out_root": str(out_root),
                "phase_filter": sorted(phase_filter) if phase_filter else None,
                "phase_task_rules": {k: sorted(v) for k, v in phase_task_rules.items()},
                "n_files": len(all_files),
                "records": records,
                "layer_group_max": best_layer,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"Wrote {summary_path} ({len(records)} records, {len(best_layer)} multi-layer groups)")


if __name__ == "__main__":
    main()
