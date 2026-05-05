"""
Convert VLMEvalKit-style .xlsx result sheets into the JSON list format expected by
`prompting_with_steering.py` / `JUDGE/eval_suite.py` (each row -> dict with at least
`model_output`, optionally `question`, `answer`, `type`, `question_id`).

VLMEvalKit column names vary by benchmark; this script picks columns by alias lists.
Override with --col-* if auto-detection fails.

Usage (from repo root):
  python JUDGE/xlsx_to_steering_json.py --input RESULT_VLMEvalKit/foo.xlsx --output out.json
  python JUDGE/xlsx_to_steering_json.py --input foo.xlsx --dry-run  # print detected columns only

Note: `.gitignore` contains `RESULT*` so RESULT_VLMEvalKit/ is not tracked unless you
`git add -f` or adjust ignore rules.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import pandas as pd

# Case-insensitive column matching (first match wins per role)
_ALIASES: Dict[str, List[str]] = {
    "model_output": [
        "prediction",
        "prediction_raw",
        "response",
        "model_output",
        "res",
        "output",
        "answer_pred",
        "pred",
    ],
    "question": ["question", "prompt", "query", "text"],
    "answer": ["answer", "gt", "ground_truth", "grouth_truth", "label", "gold"],
    "question_id": ["question_id", "id", "index", "idx", "uid", "sample_id", "image_id"],
    "type": [
        "type",
        "category",
        "sub_task",
        "task",
        "split",
        "l2-category",
        "l2_category",
        "dataset",
        "dataset_name",
        "benchmark",
        "benchmark_name",
    ],
    "source_filename": ["source_filename", "filename", "image", "image_path"],
}


def _norm_col(c: str) -> str:
    return re.sub(r"\s+", " ", str(c).strip().lower())


def _pick_column(
    df: pd.DataFrame,
    role: str,
    overrides: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    if overrides and role in overrides and overrides[role]:
        name = overrides[role]
        if name in df.columns:
            return name
        raise ValueError(f"Override column {role}={name!r} not in sheet columns: {list(df.columns)}")
    aliases = _ALIASES.get(role, [])
    inv = {_norm_col(c): c for c in df.columns}
    for a in aliases:
        key = _norm_col(a)
        if key in inv:
            return inv[key]
    return None


def xlsx_to_rows(
    path: str,
    sheet: Optional[str] = None,
    col_overrides: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    xl = pd.ExcelFile(path)
    name = sheet or xl.sheet_names[0]
    df = pd.read_excel(path, sheet_name=name, engine="openpyxl")
    df.columns = [str(c) for c in df.columns]

    pred_col = _pick_column(df, "model_output", col_overrides)
    if not pred_col:
        raise ValueError(
            "Could not find a prediction column. "
            f"Columns: {list(df.columns)}. Use --col-model_output NAME."
        )

    q_col = _pick_column(df, "question", col_overrides)
    a_col = _pick_column(df, "answer", col_overrides)
    id_col = _pick_column(df, "question_id", col_overrides)
    type_col = _pick_column(df, "type", col_overrides)
    src_col = _pick_column(df, "source_filename", col_overrides)

    rows: List[Dict[str, Any]] = []
    for i in range(len(df)):
        r = df.iloc[i]
        def _cell(col: Optional[str]) -> Any:
            if not col:
                return None
            v = r[col]
            if pd.isna(v):
                return None
            if isinstance(v, float) and v == int(v):
                return int(v)
            return v

        pred = _cell(pred_col)
        item: Dict[str, Any] = {
            "model_output": "" if pred is None else str(pred).strip(),
        }
        if q_col is not None:
            qv = _cell(q_col)
            if qv is not None:
                item["question"] = str(qv)
        if a_col is not None:
            av = _cell(a_col)
            if av is not None:
                item["answer"] = str(av).strip()
        if id_col is not None:
            iv = _cell(id_col)
            if iv is not None:
                item["question_id"] = str(iv)
        else:
            item["question_id"] = str(i)
        if type_col is not None:
            tv = _cell(type_col)
            if tv is not None:
                item["type"] = str(tv).strip()
        if src_col is not None:
            sv = _cell(src_col)
            if sv is not None:
                item["source_filename"] = str(sv)

        rows.append(item)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="VLMEvalKit xlsx -> steering-style JSON")
    p.add_argument("--input", required=True, help="path to .xlsx")
    p.add_argument("--output", default=None, help="write JSON list here (stdout if omitted)")
    p.add_argument("--sheet", default=None, help="sheet name (default: first sheet)")
    p.add_argument("--dry-run", action="store_true", help="only print detected columns and row count")
    p.add_argument("--col-model_output", default=None, dest="col_model_output")
    p.add_argument("--col-question", default=None)
    p.add_argument("--col-answer", default=None)
    p.add_argument("--col-question_id", default=None)
    p.add_argument("--col-type", default=None)
    p.add_argument("--col-source_filename", default=None)
    args = p.parse_args()

    overrides = {
        k: v
        for k, v in {
            "model_output": args.col_model_output,
            "question": args.col_question,
            "answer": args.col_answer,
            "question_id": args.col_question_id,
            "type": args.col_type,
            "source_filename": args.col_source_filename,
        }.items()
        if v
    }

    if args.dry_run:
        xl = pd.ExcelFile(args.input)
        sn = args.sheet or xl.sheet_names[0]
        df0 = pd.read_excel(args.input, sheet_name=sn, engine="openpyxl", nrows=0)
        dfn = pd.read_excel(args.input, sheet_name=sn, engine="openpyxl")
        df0.columns = [str(c) for c in df0.columns]
        print(f"sheet={sn!r} data_rows={len(dfn)}")
        print(f"xlsx_columns={list(df0.columns)}")
        try:
            rows = xlsx_to_rows(args.input, sheet=args.sheet, col_overrides=overrides or None)
            print(f"mapped_keys_example={list(rows[0].keys()) if rows else []}")
        except Exception as e:
            print(f"auto-map failed: {e}")
        return

    rows = xlsx_to_rows(args.input, sheet=args.sheet, col_overrides=overrides or None)

    text = json.dumps(rows, ensure_ascii=False, indent=2, default=str)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {len(rows)} rows -> {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
