"""
Inst-It-Bench 统一评测模块
整合了 Image MC/OE 和 Video MC/OE 的评测功能
"""
import argparse
import json
import os
import re
import glob
import csv
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from openai import OpenAI
from utils.prompts import prompt_template


# ============ 常量定义 ============

PROMPTS = {
    "image": prompt_template["gpt_eval"]["image"],
    "video": prompt_template["gpt_eval"]["video"]
}

# 元数据解析模式
META_PATTERNS = {
    "layer": re.compile(r"layer[_=]([0-9]+)"),
    "multiplier": re.compile(r"multiplier[_=]([-+0-9.]+)"),
    "behavior": re.compile(r"behavior[_=]([^_]+)"),
    "type": re.compile(r"type[_=]([a-zA-Z0-9_]+?)_model_name"),
    "model_name": re.compile(r"model_name[_=]([^_]+)"),
    "model_size": re.compile(r"model_size[_=]([a-zA-Z0-9]+)")
}


# ============ 工具函数 ============

def parse_meta_from_filename(fname: str) -> dict:
    """从文件名中解析元数据"""
    fname_no_ext = fname.replace('.json', '').replace('_evaluated', '')
    meta = {}
    for k, pat in META_PATTERNS.items():
        m = pat.search(fname_no_ext)
        if m:
            meta[k] = m.group(1)
    return meta


def extract_answer(text: str) -> str:
    """从模型输出中提取答案选项 (A/B/C/D)"""
    if not text:
        return ""
    
    text = text.strip()
    
    # 直接匹配单个字母
    if text.upper() in ["A", "B", "C", "D"]:
        return text.upper()
    
    # 匹配开头的字母
    match = re.match(r'^([A-Da-d])[.\s\)]', text)
    if match:
        return match.group(1).upper()
    
    # 匹配 "The answer is X" 格式
    match = re.search(r'(?:answer|option|choice)\s*(?:is|:)?\s*([A-Da-d])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # 匹配文本中的第一个单独的 A/B/C/D
    match = re.search(r'\b([A-Da-d])\b', text)
    if match:
        return match.group(1).upper()
    
    return text.strip()[:1].upper() if text else ""


def extract_characters_regex(s: str) -> str:
    """处理多选题响应中的前缀，提取答案字符（A/B/C/D）"""
    s = s.strip()
    answer_prefixes = [
        "The best answer is", "The correct answer is", "The answer is",
        "The answer", "The best option is", "The correct option is",
        "Best answer:", "Best option:", "Answer", "Answer:", "answer:", "answer",
        "Option:", "The correct answer", "The correct option",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    m = re.search(r'[ABCD]', s)
    return m[0] if m else ""


def extract_actual_question(full_question: str) -> str:
    """从完整的 prompt 中提取实际的问题部分"""
    # 尝试提取 "please answer the question:" 后面的内容
    patterns = [
        r"please answer the question:\s*\n?(.*?)$",
        r"please answer the question:\s*(.*?)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, full_question, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return full_question.strip()


def index_gt_by_question_id(gt_list: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """Map question_id -> ground-truth row (for mixed / non-contiguous prediction lists)."""
    out: Dict[str, Dict[str, str]] = {}
    for g in gt_list:
        qid = str(g.get("question_id", "") or "").strip()
        if qid:
            out[qid] = g
    return out


# ============ 数据集加载 ============

def load_inst_it_image_mc_ground_truth() -> List[Dict[str, str]]:
    """加载 Inst-It-Bench image_multi_choice 数据集"""
    print("Loading Inst-It-Bench image_multi_choice dataset for ground truth...")
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        
        parquet_path = hf_hub_download(
            repo_id='Inst-IT/Inst-It-Bench',
            filename='image_multi_choice/test-00000-of-00001.parquet',
            repo_type='dataset'
        )
        
        df = pd.read_parquet(parquet_path)
        
        gt_list = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            gt_list.append({
                "question_id": str(row.get("question_id", f"{idx:03d}-{idx:02d}")),
                "question": row.get("question", "").strip(),
                "answer": row.get("answer", ""),  # 正确答案选项 (A/B/C/D)
            })
        
        print(f"  Loaded {len(gt_list)} ground truth entries from Inst-It-Bench image_multi_choice")
        return gt_list
    except Exception as e:
        print(f"  Warning: Failed to load Inst-It-Bench dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


def load_inst_it_video_mc_ground_truth() -> List[Dict[str, str]]:
    """加载 Inst-It-Bench video_multi_choice 数据集"""
    print("Loading Inst-It-Bench video_multi_choice dataset for ground truth...")
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        
        parquet_path = hf_hub_download(
            repo_id='Inst-IT/Inst-It-Bench',
            filename='video_multi_choice/test-00000-of-00001.parquet',
            repo_type='dataset'
        )
        
        df = pd.read_parquet(parquet_path)
        
        gt_list = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            gt_list.append({
                "question_id": str(row.get("question_id", f"{idx:03d}-{idx:02d}")),
                "question": row.get("question", "").strip(),
                "answer": row.get("answer", ""),  # 正确答案选项 (A/B/C/D)
            })
        
        print(f"  Loaded {len(gt_list)} ground truth entries from Inst-It-Bench video_multi_choice")
        return gt_list
    except Exception as e:
        print(f"  Warning: Failed to load Inst-It-Bench dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


def load_inst_it_image_oe_ground_truth() -> List[Dict[str, str]]:
    """加载 Inst-It-Bench image_open_ended 数据集"""
    print("Loading Inst-It-Bench image_open_ended dataset for ground truth...")
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        
        parquet_path = hf_hub_download(
            repo_id='Inst-IT/Inst-It-Bench',
            filename='image_open_ended/test-00000-of-00001.parquet',
            repo_type='dataset'
        )
        
        df = pd.read_parquet(parquet_path)
        
        gt_list = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            gt_list.append({
                "question_id": str(row.get("question_id", "")),
                "question": row.get("question", "").strip(),
                "answer": row.get("answer", ""),
            })
        
        print(f"  Loaded {len(gt_list)} ground truth entries from Inst-It-Bench image_open_ended")
        return gt_list
    except Exception as e:
        print(f"  Warning: Failed to load Inst-It-Bench dataset: {e}")
        return []


def load_inst_it_video_oe_ground_truth() -> List[Dict[str, str]]:
    """加载 Inst-It-Bench video_open_ended 数据集"""
    print("Loading Inst-It-Bench video_open_ended dataset for ground truth...")
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        
        parquet_path = hf_hub_download(
            repo_id='Inst-IT/Inst-It-Bench',
            filename='video_open_ended/test-00000-of-00001.parquet',
            repo_type='dataset'
        )
        
        df = pd.read_parquet(parquet_path)
        
        gt_list = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            gt_list.append({
                "question_id": str(row.get("question_id", "")),
                "question": row.get("question", "").strip(),
                "answer": row.get("answer", ""),
            })
        
        print(f"  Loaded {len(gt_list)} ground truth entries from Inst-It-Bench video_open_ended")
        return gt_list
    except Exception as e:
        print(f"  Warning: Failed to load Inst-It-Bench dataset: {e}")
        return []


# ============ vLLM Judge ============

class VLLMJudge:
    """使用 vLLM 进行评分的 Judge 类"""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1", model_name: str = None, timeout: int = 300):
        print(f"Connecting to vLLM service at: {base_url}")
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)

        if model_name is None:
            try:
                models = self.client.models.list()
                self.model_name = models.data[0].id
                print(f"Auto-detected model: {self.model_name}")
            except Exception as e:
                print(f"Warning: Could not auto-detect model: {e}")
                self.model_name = "Qwen/Qwen2.5-72B-Instruct-AWQ"
        else:
            self.model_name = model_name

        print(f"Using model: {self.model_name}")

    def judge(self, question: str, model_output: str, ground_truth: str = "", split: str = "image") -> Tuple[float, str]:
        """对单个样本进行评分，返回: (score, raw_response)"""
        prompt = PROMPTS.get(split, PROMPTS["image"])
        eval_input = json.dumps({
            "question": question,
            "ground_truth_answer": ground_truth,
            "tester_response": model_output
        }, ensure_ascii=False)
        
        full_prompt = (
            f"{prompt}\n\n"
            f"{eval_input}\n\n"
            "Please provide a score between 0 and 1, where 0 means completely incorrect and 1 means completely correct. "
            "Output ONLY a single number between 0 and 1."
        )

        messages = [{"role": "user", "content": full_prompt}]
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=64,
            top_p=1.0,
        )
        text = resp.choices[0].message.content.strip()
        score = self._extract_score(text)
        return score, text

    @staticmethod
    def _extract_score(text: str) -> float:
        """从响应文本中提取分数"""
        for pat in [r"\b([01](?:\.\d+)?)\b", r"\b(0?\.\d+)\b"]:
            m = re.findall(pat, text)
            if m:
                try:
                    v = float(m[0])
                    if 0.0 <= v <= 1.0:
                        return v
                except Exception:
                    pass
        try:
            v = float(text)
            if 0.0 <= v <= 1.0:
                return v
        except Exception:
            pass
        print(f"[Warn] cannot extract score from: {text!r}")
        return 0.0


# ============ MC 评测函数 ============

def evaluate_mc_single_prediction(
    item: dict,
    idx: int,
    gt_list: List[Dict[str, str]],
    split: str = "image",
    pre_extracted: Optional[str] = None,
    gt_by_qid: Optional[Dict[str, Dict[str, str]]] = None,
) -> dict:
    """评测单个 MC 预测。若提供 pre_extracted（如 think 任务经 LLM 抽取的 A–D），则不再做正则提取。"""
    full_question = item.get("question", "")
    model_output = item.get("model_output", "") or item.get("raw_model_output", "")
    
    # 从 ground truth 列表获取正确答案（优先 question_id 对齐，便于同一 xlsx 内混合子集）
    question_id = ""
    ground_truth = ""
    gt_entry: Dict[str, str] = {}
    qid = str(item.get("question_id", "") or "").strip()
    if gt_by_qid is not None:
        if qid and qid in gt_by_qid:
            gt_entry = gt_by_qid[qid]
        elif qid:
            # 有 id 但不在 GT 表里（不应再按 idx 错位对齐）
            gt_entry = {}
        elif gt_list and idx < len(gt_list):
            gt_entry = gt_list[idx]
    elif gt_list and idx < len(gt_list):
        gt_entry = gt_list[idx]
    if gt_entry:
        question_id = str(gt_entry.get("question_id", qid or f"{idx:03d}-{idx:02d}"))
        ground_truth = gt_entry.get("answer", "")
    
    # 提取模型预测的答案
    if pre_extracted is not None:
        pe = (pre_extracted or "").strip().upper()
        extracted_prediction = pe if pe in ("A", "B", "C", "D") else ""
    elif split == "image":
        extracted_prediction = extract_characters_regex(model_output)
    else:  # video
        extracted_prediction = extract_answer(model_output)
    
    # 判断是否正确
    is_correct = extracted_prediction.upper() == ground_truth.upper() if ground_truth else False
    
    return {
        "question_id": question_id,
        "question": extract_actual_question(full_question),
        "model_prediction": model_output,
        "ground_truth": ground_truth,
        "split": split,
        "extracted_model_prediction": extracted_prediction,
        "is_correct": is_correct,
    }


def evaluate_mc_file(
    input_file: str,
    output_file: str,
    gt_list: List[Dict[str, str]],
    split: str = "image",
    *,
    think_extract: bool = False,
    match_by_question_id: bool = True,
    base_url: str = "http://127.0.0.1:23333/v1",
    model_name: str = "Qwen/Qwen2.5-72B-Instruct-AWQ",
    num_workers: int = 8,
    api_key: str = "EMPTY",
) -> dict:
    """评测单个 MC JSON 文件并保存结果。think_extract=True 时用 LLM（mc_think_extract）从含推理的输出中抽取选项。
    默认 match_by_question_id=True：按 question_id 与官方 GT 对齐；条目数少于 GT 时不会错位。
    无 question_id 的条目仍按列表下标回退（兼容旧 JSON）。"""
    from mc_think_extract import ThinkMCAnswerExtractor

    with open(input_file, "r", encoding="utf-8") as f:
        preds = json.load(f)

    gt_by_qid = index_gt_by_question_id(gt_list) if match_by_question_id else None

    pre_extracted_list: Optional[List[Optional[str]]] = None
    if think_extract:
        extractor = ThinkMCAnswerExtractor(
            base_url=base_url, model_name=model_name, api_key=api_key
        )
        n = len(preds)
        pre_extracted_list = [""] * n
        if n > 0:

            def _one(i: int) -> Tuple[int, str]:
                text = preds[i].get("model_output", "") or preds[i].get("raw_model_output", "") or ""
                return i, extractor.extract(text)

            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futs = [ex.submit(_one, i) for i in range(n)]
                for fut in tqdm(
                    as_completed(futs),
                    total=n,
                    desc=f"Think MC extract {os.path.basename(input_file)}",
                    leave=False,
                ):
                    i, letter = fut.result()
                    pre_extracted_list[i] = letter
    
    total_num = len(preds)
    valid_num = 0
    correct_num = 0
    results = []
    
    for idx, item in enumerate(tqdm(preds, desc=f"Evaluating {os.path.basename(input_file)}", leave=False)):
        pe = pre_extracted_list[idx] if pre_extracted_list is not None else None
        result = evaluate_mc_single_prediction(
            item, idx, gt_list, split, pre_extracted=pe, gt_by_qid=gt_by_qid
        )
        results.append(result)
        
        if result["ground_truth"]:
            valid_num += 1
            if result["is_correct"]:
                correct_num += 1
    
    accuracy = (correct_num / valid_num * 100.0) if valid_num > 0 else 0.0
    
    output_data = {
        "accuracy": accuracy,
        "total_num": total_num,
        "valid_num": valid_num,
        "correct_num": correct_num,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mc_extract": "think_llm" if think_extract else "regex",
        "model_predictons": [
            {
                "question_id": r["question_id"],
                "question": r["question"],
                "model_prediction": r["model_prediction"],
                "ground_truth": r["ground_truth"],
                "split": r["split"],
                "extracted_model_prediction": r["extracted_model_prediction"],
            }
            for r in results
        ],
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    return {
        "accuracy": accuracy,
        "total_num": total_num,
        "valid_num": valid_num,
        "correct_num": correct_num,
    }


# ============ OE 评测函数 ============

def judge_oe_single_item(
    judge: VLLMJudge, 
    item: dict, 
    item_idx: int,
    gt_list: List[Dict[str, str]] = None,
    split: str = "image",
    gt_by_qid: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[dict, float, bool]:
    """评分单个 OE 数据项"""
    full_question = item.get("question", "")
    model_output = item.get("model_output", "") or item.get("raw_model_output", "")
    
    # 优先使用数据中已有的 ground_truth
    ground_truth = item.get("ground_truth", "") or item.get("answer", "")
    question_id = item.get("question_id", "")
    
    # 如果数据中没有 ground_truth，按 question_id 或索引从 gt_list 查找
    if not ground_truth:
        qid = str(question_id or "").strip()
        if gt_by_qid is not None:
            if qid and qid in gt_by_qid:
                ground_truth = gt_by_qid[qid].get("answer", "")
                question_id = gt_by_qid[qid].get("question_id", question_id)
            elif qid:
                pass
            elif gt_list and item_idx < len(gt_list):
                gt_entry = gt_list[item_idx]
                ground_truth = gt_entry.get("answer", "")
                question_id = gt_entry.get("question_id", "")
        elif gt_list and item_idx < len(gt_list):
            gt_entry = gt_list[item_idx]
            ground_truth = gt_entry.get("answer", "")
            question_id = gt_entry.get("question_id", "")
    
    try:
        score, raw_response = judge.judge(full_question, model_output, ground_truth, split)
        
        result_item = {
            "question_id": question_id,
            "question": full_question,
            "model_output": item.get("model_output", ""),
            "raw_model_output": item.get("raw_model_output", ""),
            "ground_truth": ground_truth,
            "score": score,
            "evaluator_response": raw_response,
            "eval_error": None
        }
        
        return result_item, score, True
    except Exception as e:
        print(f"[Error] Failed to judge item {item_idx}: {e}")
        result_item = {
            "question_id": question_id,
            "question": full_question,
            "model_output": item.get("model_output", ""),
            "raw_model_output": item.get("raw_model_output", ""),
            "ground_truth": ground_truth,
            "score": 0.0,
            "evaluator_response": None,
            "eval_error": str(e)
        }
        return result_item, 0.0, False


def process_oe_json_file(
    input_path: str,
    output_path: str,
    judge: VLLMJudge,
    num_workers: int = 8,
    overwrite: bool = False,
    gt_list: List[Dict[str, str]] = None,
    split: str = "image",
    match_by_question_id: bool = True,
) -> dict:
    """处理单个 OE JSON 文件。默认按 question_id 对齐 GT，避免预测条数少于官方时错位判分。"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print(f"[Warn] Expected list, got {type(data)} in {input_path}")
        return {"error": "Invalid data format"}

    gt_by_qid = index_gt_by_question_id(gt_list) if (match_by_question_id and gt_list) else None
    
    # 统计需要评分的数据
    items_to_judge = []
    items_skipped = 0
    for idx, item in enumerate(data):
        if "score" in item and not overwrite:
            items_skipped += 1
        else:
            items_to_judge.append((idx, item))
    
    print(f"  Total items: {len(data)}, To judge: {len(items_to_judge)}, Skipped (has score): {items_skipped}")
    
    if not items_to_judge:
        print(f"  All items already have scores, skipping...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {
            "total": len(data),
            "judged": 0,
            "skipped": items_skipped,
            "avg_score": None
        }
    
    # 并发评分
    total_score = 0.0
    valid_count = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 使用字典来跟踪 future 和原始索引的映射
        future_to_idx = {}
        futures = []
        for idx, item in items_to_judge:
            future = executor.submit(
                judge_oe_single_item, judge, item, idx, gt_list, split, gt_by_qid
            )
            future_to_idx[future] = idx
            futures.append(future)
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Judging", leave=False):
            item, score, success = future.result()
            original_idx = future_to_idx[future]
            data[original_idx] = item
            if success:
                total_score += score
                valid_count += 1
    
    # 计算平均分
    avg_score = total_score / valid_count if valid_count > 0 else 0.0
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return {
        "total": len(data),
        "judged": len(items_to_judge),
        "skipped": items_skipped,
        "avg_score": avg_score,
        "valid_count": valid_count,
        "total_score": total_score
    }


# ============ 批量评测函数 ============

def bulk_evaluate_mc(
    input_dir: str,
    output_dir: str,
    pattern: str,
    summary_out: Optional[str],
    gt_list: List[Dict[str, str]],
    split: str = "image",
    layer_range: Optional[Tuple[int, int]] = None,
    *,
    think_extract: bool = False,
    base_url: str = "http://127.0.0.1:23333/v1",
    model_name: str = "Qwen/Qwen2.5-72B-Instruct-AWQ",
    num_workers: int = 8,
    api_key: str = "EMPTY",
):
    """批量评测 MC 格式的文件"""
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        print(f"[Warning] No files matched: {os.path.join(input_dir, pattern)}")
        return []

    # 过滤文件：只保留指定层范围内的文件
    if layer_range:
        filtered_files = []
        for fp in files:
            meta = parse_meta_from_filename(os.path.basename(fp))
            layer = int(meta.get("layer", -1))
            if layer_range[0] <= layer <= layer_range[1]:
                filtered_files.append(fp)
        files = filtered_files
        print(f"Filtered to {len(files)} files in layer range {layer_range[0]}-{layer_range[1]}")

    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for fp in tqdm(files, desc=f"Bulk evaluating {input_dir}"):
        base = os.path.basename(fp)
        output_file = os.path.join(output_dir, base)
        
        res = evaluate_mc_file(
            input_file=fp,
            output_file=output_file,
            gt_list=gt_list,
            split=split,
            think_extract=think_extract,
            base_url=base_url,
            model_name=model_name,
            num_workers=num_workers,
            api_key=api_key,
        )
        
        meta = parse_meta_from_filename(base)

        rows.append({
            "file": base,
            "layer": meta.get("layer", ""),
            "multiplier": meta.get("multiplier", ""),
            "behavior": meta.get("behavior", ""),
            "type": meta.get("type", ""),
            "model_name": meta.get("model_name", ""),
            "model_size": meta.get("model_size", ""),
            "accuracy": res["accuracy"],
            "valid_num": res["valid_num"],
            "correct_num": res["correct_num"],
            "total_num": res["total_num"],
        })
        
        print(f"  -> {base}: accuracy={res['accuracy']:.2f}%, correct={res['correct_num']}/{res['valid_num']}")

    if summary_out and rows:
        os.makedirs(os.path.dirname(summary_out), exist_ok=True)
        with open(summary_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file", "layer", "multiplier", "behavior", "type", "model_name", "model_size",
                    "accuracy", "valid_num", "correct_num", "total_num",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Summary CSV] {summary_out}  ({len(rows)} runs)")
    
    return rows


def bulk_evaluate_oe(
    input_dir: str,
    output_dir: str,
    pattern: str,
    summary_out: Optional[str],
    judge: VLLMJudge,
    gt_list: List[Dict[str, str]],
    split: str = "image",
    num_workers: int = 8,
    overwrite: bool = False,
    layer_range: Optional[Tuple[int, int]] = None,
):
    """批量评测 OE 格式的文件"""
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        print(f"[Warning] No files matched: {os.path.join(input_dir, pattern)}")
        return []

    # 过滤文件
    if layer_range:
        filtered_files = []
        for fp in files:
            meta = parse_meta_from_filename(os.path.basename(fp))
            layer = int(meta.get("layer", -1))
            if layer_range[0] <= layer <= layer_range[1]:
                filtered_files.append(fp)
        files = filtered_files
        print(f"Filtered to {len(files)} files in layer range {layer_range[0]}-{layer_range[1]}")

    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for fp in tqdm(files, desc=f"Bulk evaluating {input_dir}"):
        base = os.path.basename(fp)
        # 如果输入在 01_raw 文件夹，输出到同级的 02_judged 文件夹
        if "/01_raw/" in fp:
            output_file = fp.replace("/01_raw/", "/02_judged/").replace(".json", "_evaluated.json")
        else:
            output_file = os.path.join(output_dir, base.replace(".json", "_evaluated.json"))
        
        res = process_oe_json_file(
            input_path=fp,
            output_path=output_file,
            judge=judge,
            num_workers=num_workers,
            overwrite=overwrite,
            gt_list=gt_list,
            split=split
        )
        
        if "error" in res:
            continue
        
        meta = parse_meta_from_filename(base)
        avg_score = res.get("avg_score")
        accuracy = avg_score * 100 if avg_score is not None else 0.0

        rows.append({
            "file": base,
            "layer": meta.get("layer", ""),
            "multiplier": meta.get("multiplier", ""),
            "behavior": meta.get("behavior", ""),
            "type": meta.get("type", ""),
            "model_name": meta.get("model_name", ""),
            "model_size": meta.get("model_size", ""),
            "accuracy": accuracy,
            "valid_num": res.get("valid_count", 0),
            "total_num": res.get("total", 0),
            "avg_score": avg_score if avg_score is not None else 0.0,
        })
        
        if avg_score is not None:
            print(f"  -> {base}: avg_score={avg_score:.4f}")

    if summary_out and rows:
        os.makedirs(os.path.dirname(summary_out), exist_ok=True)
        with open(summary_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file", "layer", "multiplier", "behavior", "type", "model_name", "model_size",
                    "accuracy", "valid_num", "total_num", "avg_score",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Summary CSV] {summary_out}  ({len(rows)} runs)")
    
    return rows


# ============ 主程序入口 ============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Inst-It-Bench evaluation tool")
    
    # 评测类型
    parser.add_argument("--eval_type", type=str, 
                       choices=["image_mc", "image_think_mc", "image_oe", "video_mc", "video_think_mc", "video_oe"], 
                       required=True,
                       help="Evaluation type (image_think_mc / video_think_mc: LLM extracts MC letter)")
    
    # 目录参数
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input directory containing JSON files (usually 01_raw)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: 02_judged for OE, same as input for MC)")
    parser.add_argument("--pattern", type=str, default=None,
                       help="Glob pattern to match files (auto-detected if not provided)")
    parser.add_argument("--summary_out", type=str, default=None,
                       help="CSV file path to save summary")
    
    # vLLM 相关参数（用于 OE）
    parser.add_argument("--base_url", type=str, default="http://localhost:8001/v1",
                       help="vLLM OpenAI API endpoint")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-72B-Instruct-AWQ",
                       help="Judge model name for OE")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of parallel workers for OE")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing scores")
    
    # 层范围参数
    parser.add_argument("--layer_range", nargs=2, type=int, default=None,
                       help="Only evaluate layers in this range (e.g., 0 35)")
    
    # 其他参数
    parser.add_argument("--no_load_gt", action="store_true",
                       help="Do not load ground truth from Inst-It-Bench dataset")

    args = parser.parse_args()

    # 如果没有提供 pattern，根据 eval_type 自动设置
    if args.pattern is None:
        if args.eval_type == "image_mc":
            args.pattern = "results_layer_*_multiplier_*_behavior_refer_type_inst_it_image_mc_qa_model_name_*_model_size_*.json"
        elif args.eval_type == "image_think_mc":
            args.pattern = "results_layer_*_multiplier_*_behavior_refer_type_inst_it_image_think_mc_qa_model_name_*_model_size_*.json"
        elif args.eval_type == "image_oe":
            args.pattern = "results_layer_*_multiplier_*_behavior_refer_type_inst_it_image_oe_qa_model_name_*_model_size_*.json"
        elif args.eval_type == "video_mc":
            args.pattern = "results_layer_*_multiplier_*_behavior_refer_type_inst_it_video_mc_qa_model_name_*_model_size_*.json"
        elif args.eval_type == "video_think_mc":
            args.pattern = "results_layer_*_multiplier_*_behavior_refer_type_inst_it_video_think_mc_qa_model_name_*_model_size_*.json"
        else:  # video_oe
            args.pattern = "results_layer_*_multiplier_*_behavior_refer_type_inst_it_video_oe_qa_model_name_*_model_size_*.json"

    # 设置输出目录
    if args.output_dir is None:
        if args.eval_type.endswith("_oe"):
            # OE 格式：如果输入在 01_raw，输出到 02_judged
            if "/01_raw/" in args.input_dir or args.input_dir.endswith("01_raw"):
                args.output_dir = args.input_dir.replace("/01_raw/", "/02_judged/").replace("01_raw", "02_judged")
            else:
                args.output_dir = os.path.join(args.input_dir, "02_judged")
        else:
            # MC 格式：输出到同目录
            args.output_dir = args.input_dir

    # 设置 summary_out 默认路径（与 output_dir 同级）
    if args.summary_out is None:
        output_parent = os.path.dirname(args.output_dir)
        if not output_parent:  # 如果 output_dir 是根目录，使用当前目录
            output_parent = "."
        args.summary_out = os.path.join(output_parent, "summary.csv")

    # 加载 ground truth
    gt_list = None
    if not args.no_load_gt:
        if args.eval_type == "image_mc":
            gt_list = load_inst_it_image_mc_ground_truth()
        elif args.eval_type == "image_think_mc":
            gt_list = load_inst_it_image_mc_ground_truth()
        elif args.eval_type == "image_oe":
            gt_list = load_inst_it_image_oe_ground_truth()
        elif args.eval_type == "video_mc":
            gt_list = load_inst_it_video_mc_ground_truth()
        elif args.eval_type == "video_think_mc":
            gt_list = load_inst_it_video_mc_ground_truth()
        else:  # video_oe
            gt_list = load_inst_it_video_oe_ground_truth()

    # 根据评测类型调用相应的函数
    if args.eval_type.endswith("_mc"):
        # MC 评测
        split = "image" if args.eval_type.startswith("image") else "video"
        think_extract = args.eval_type in ("image_think_mc", "video_think_mc")
        bulk_evaluate_mc(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            summary_out=args.summary_out,
            gt_list=gt_list or [],
            split=split,
            layer_range=tuple(args.layer_range) if args.layer_range else None,
            think_extract=think_extract,
            base_url=args.base_url,
            model_name=args.model_name,
            num_workers=args.num_workers,
        )
    else:
        # OE 评测
        split = "image" if args.eval_type.startswith("image") else "video"
        judge = VLLMJudge(base_url=args.base_url, model_name=args.model_name)
        bulk_evaluate_oe(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            summary_out=args.summary_out,
            judge=judge,
            gt_list=gt_list or [],
            split=split,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
            layer_range=tuple(args.layer_range) if args.layer_range else None,
        )

    print("Done.")
