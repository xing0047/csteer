import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

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
    raw = _load_json_list_or_dict(results_path)
    fallback_preds: List[str] = []
    if isinstance(raw, dict):
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

            rec = grade_results.setdefault(sample_id, {"model": [], "content": [], "score": []})
            rec["model"].append(model_name)
            rec["content"].append(content)
            rec["score"].append(score)

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
        to_save = {k: v for k, v in summary.items() if k != "grade_results"}
        to_save["grade_results"] = grade_results
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
        print(f"[vip] wrote {out_path}")
    print(df)
    return summary
