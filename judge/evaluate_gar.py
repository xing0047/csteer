"""
GAR 统一评测模块
"""
import argparse
import json
import os
import re
import glob
import csv
from datetime import datetime
from tqdm import tqdm
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from utils.prompts import prompt_template


# ============ 常量定义 ============

# GAR-detailed 和 GAR-simple 评测 prompt
GAR_DETAILED_PROMPT = prompt_template["gar_detailed_eval"]
GAR_SIMPLE_PROMPT = prompt_template["gar_simple_eval"]

# 类型映射：将JSON中的type值映射到图片中的官方类型名称（用于 VQA）
TYPE_MAPPING = {
    "color": "Color",
    "shape": "Shape",
    "texture/pattern": "Texture",
    "material": "Material",
    "ordering": "Position",
    "mirror": "Non-Entity",
    "relation": "Relation"
}

# 元数据解析模式
META_PATTERNS = {
    "layer": re.compile(r"layer[_=]([0-9]+)"),
    "multiplier": re.compile(r"multiplier[_=]([-+0-9.]+)"),
    "behavior": re.compile(r"behavior[_=]([^_]+)"),
    "type": re.compile(r"type[_=]([^_]+)"),
    "model_name": re.compile(r"model_name[_=]([^_]+)"),
    "model_size": re.compile(r"model_size[_=]([^_]+)")
}


# ============ 工具函数 ============

def extract_characters_regex(s: str) -> str:
    """
    处理多选题响应中的前缀，提取答案字符（A/B/C/D）
    从 VideoMME 借鉴并简化
    """
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


def parse_meta_from_filename(fname: str) -> dict:
    """
    从文件名中解析元数据（layer, multiplier, behavior, type, model_name, model_size）
    """
    meta = {}
    for k, pat in META_PATTERNS.items():
        m = pat.search(fname)
        if m:
            meta[k] = m.group(1)
    return meta


def _extract_boolean(text: str) -> bool:
    """
    从文本中提取 True/False
    """
    text_lower = text.strip().lower()
    if "true" in text_lower:
        return True
    elif "false" in text_lower:
        return False
    # 默认返回 False
    print(f"[Warn] cannot extract boolean from: {text!r}, defaulting to False")
    return False


# ============ 评测器类 ============

class VLLMEvaluator:
    """vLLM OpenAI-compatible evaluator"""
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


# ============ GAR-Detailed 评测 ============

def evaluate_prediction_gar_detailed(vllm_evaluator: VLLMEvaluator, model_pred: dict):
    """
    评测 GAR-detailed 格式的预测，返回 True/False
    
    Args:
        vllm_evaluator: VLLMEvaluator 实例
        model_pred: 包含 subject_name, object_name, predicate_name, model_output 的字典
    
    Returns:
        (model_pred, is_true, is_valid): 更新后的预测字典、是否为True、是否有效
    """
    eval_input = json.dumps({
        "subject_name": model_pred.get("subject_name", ""),
        "object_name": model_pred.get("object_name", ""),
        "predicate_name": model_pred.get("predicate_name", ""),
        "model_output": model_pred.get("model_output", "")
    }, ensure_ascii=False)
    
    try:
        # 使用 vLLM 评测，但需要特殊处理返回的 True/False
        full_prompt = (
            f"{GAR_DETAILED_PROMPT}\n\n"
            f"{eval_input}\n\n"
            "Please output ONLY 'True' or 'False' (without quotes, without any other text)."
        )
        
        messages = [{"role": "user", "content": full_prompt}]
        resp = vllm_evaluator.client.chat.completions.create(
            model=vllm_evaluator.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=10,
            top_p=1.0,
        )
        text = resp.choices[0].message.content.strip()
        
        # 提取 True/False
        is_true = _extract_boolean(text)
        model_pred["eval_result"] = "True" if is_true else "False"
        model_pred["valid"] = True
        model_pred["evaluator_response"] = text
        model_pred["error_in_eval"] = "No Error."
        return model_pred, is_true, True
    except Exception as e:
        model_pred["eval_result"] = "False"
        model_pred["valid"] = False
        model_pred["evaluator_response"] = ""
        model_pred["error_in_eval"] = str(e)
        return model_pred, False, False


def compute_metric_gar_detailed(
    model_predictions: list,
    log_path: Optional[str],
    base_url: str,
    model_name: str,
    num_process: int = 8
):
    """
    计算 GAR-detailed 格式的评测指标（True/False 准确率）
    
    Args:
        model_predictions: 预测结果列表
        log_path: 保存详细结果的路径（可选）
        base_url: vLLM 服务地址
        model_name: 评测模型名称
        num_process: 并行处理数量
    
    Returns:
        包含准确率、总数、有效数、True数等信息的字典
    """
    vllm = VLLMEvaluator(base_url=base_url, model_name=model_name)
    
    total_num = 0
    valid_num = 0
    true_num = 0
    out = []
    
    with ThreadPoolExecutor(max_workers=num_process) as ex:
        futures = [ex.submit(evaluate_prediction_gar_detailed, vllm, mp) for mp in model_predictions]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluate GAR-Detailed Predictions (vLLM):"):
            mp, is_true, ok = fut.result()
            out.append(mp)
            if ok:
                valid_num += 1
                if is_true:
                    true_num += 1
            total_num += 1
    
    accuracy = (true_num / valid_num * 100.0) if valid_num > 0 else "No valid evaluations."
    result = {
        "accuracy": accuracy,
        "total_num": total_num,
        "valid_num": valid_num,
        "true_num": true_num,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "evaluator_model": model_name,
        "evaluator_base_url": base_url,
        "eval_prompt": GAR_DETAILED_PROMPT,
        "model_predictons": out,
    }
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
    return result


# ============ GAR-Simple 评测 ============

def evaluate_prediction_gar_simple(vllm_evaluator: VLLMEvaluator, model_pred: dict):
    """
    评测 GAR-simple 格式的预测，返回 True/False
    
    Args:
        vllm_evaluator: VLLMEvaluator 实例
        model_pred: 包含 answer, model_output 的字典
    
    Returns:
        (model_pred, is_true, is_valid): 更新后的预测字典、是否为True、是否有效
    """
    eval_input = json.dumps({
        "answer": model_pred.get("answer", ""),
        "model_output": model_pred.get("model_output", "")
    }, ensure_ascii=False)
    
    try:
        # 使用 vLLM 评测，但需要特殊处理返回的 True/False
        full_prompt = (
            f"{GAR_SIMPLE_PROMPT}\n\n"
            f"{eval_input}\n\n"
            "Please output ONLY 'True' or 'False' (without quotes, without any other text)."
        )
        
        messages = [{"role": "user", "content": full_prompt}]
        resp = vllm_evaluator.client.chat.completions.create(
            model=vllm_evaluator.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=10,
            top_p=1.0,
        )
        text = resp.choices[0].message.content.strip()
        
        # 提取 True/False
        is_true = _extract_boolean(text)
        model_pred["eval_result"] = "True" if is_true else "False"
        model_pred["valid"] = True
        model_pred["evaluator_response"] = text
        model_pred["error_in_eval"] = "No Error."
        return model_pred, is_true, True
    except Exception as e:
        model_pred["eval_result"] = "False"
        model_pred["valid"] = False
        model_pred["evaluator_response"] = ""
        model_pred["error_in_eval"] = str(e)
        return model_pred, False, False


def compute_metric_gar_simple(
    model_predictions: list,
    log_path: Optional[str],
    base_url: str,
    model_name: str,
    num_process: int = 8
):
    """
    计算 GAR-simple 格式的评测指标（True/False 准确率）
    
    Args:
        model_predictions: 预测结果列表
        log_path: 保存详细结果的路径（可选）
        base_url: vLLM 服务地址
        model_name: 评测模型名称
        num_process: 并行处理数量
    
    Returns:
        包含准确率、总数、有效数、True数等信息的字典
    """
    vllm = VLLMEvaluator(base_url=base_url, model_name=model_name)
    
    total_num = 0
    valid_num = 0
    true_num = 0
    out = []
    
    with ThreadPoolExecutor(max_workers=num_process) as ex:
        futures = [ex.submit(evaluate_prediction_gar_simple, vllm, mp) for mp in model_predictions]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluate GAR-Simple Predictions (vLLM):"):
            mp, is_true, ok = fut.result()
            out.append(mp)
            if ok:
                valid_num += 1
                if is_true:
                    true_num += 1
            total_num += 1
    
    accuracy = (true_num / valid_num * 100.0) if valid_num > 0 else "No valid evaluations."
    result = {
        "accuracy": accuracy,
        "total_num": total_num,
        "valid_num": valid_num,
        "true_num": true_num,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "evaluator_model": model_name,
        "evaluator_base_url": base_url,
        "eval_prompt": GAR_SIMPLE_PROMPT,
        "model_predictons": out,
    }
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
    return result


# ============ GAR-VQA (MC) 评测 ============

def compute_metric_mc_gar_vqa(
    model_predictions,
    log_path: Optional[str],
    *,
    think_extract: bool = False,
    base_url: str = "http://127.0.0.1:23333/v1",
    model_name: str = "Qwen/Qwen2.5-72B-Instruct-AWQ",
    num_process: int = 8,
    api_key: str = "EMPTY",
):
    """
    计算 GAR VQA 格式的评测指标（True/False 准确率），支持按类型分类统计
    
    Args:
        model_predictions: 预测结果列表
        log_path: 保存详细结果的路径（可选）
        think_extract: 为 True 时用 LLM 从含推理的输出中抽取 A/B/C/D（gar_image_think_mc_qa）
    
    Returns:
        包含准确率、总数、有效数、正确数以及各类型统计信息的字典
    """
    from mc_think_extract import ThinkMCAnswerExtractor

    n = len(model_predictions)
    pre_extracted: Optional[List[str]] = None
    if think_extract and n > 0:
        extractor = ThinkMCAnswerExtractor(
            base_url=base_url, model_name=model_name, api_key=api_key
        )

        def _one(i: int) -> Tuple[int, str]:
            mp = model_predictions[i]
            pred = mp.get("model_prediction") or mp.get("model_output", "") or ""
            return i, extractor.extract(pred)

        pre_extracted = [""] * n
        with ThreadPoolExecutor(max_workers=num_process) as ex:
            futs = [ex.submit(_one, i) for i in range(n)]
            for fut in tqdm(
                as_completed(futs),
                total=n,
                desc="GAR think MC extract",
                leave=False,
            ):
                i, letter = fut.result()
                pre_extracted[i] = letter

    total_num = 0
    valid_num = 0
    correct_num = 0
    
    # 按类型分类统计
    type_stats = {}
    for type_name in TYPE_MAPPING.values():
        type_stats[type_name] = {
            "total_num": 0,
            "valid_num": 0,
            "correct_num": 0
        }

    for idx, model_pred in enumerate(
        tqdm(model_predictions, desc="Evaluate GAR VQA Multi-Choice Predictions:")
    ):
        # 支持两种字段名格式：model_prediction/model_output, ground_truth/answer
        pred = model_pred.get("model_prediction") or model_pred.get("model_output", "")
        if pre_extracted is not None:
            extracted_pred = pre_extracted[idx] or ""
        else:
            extracted_pred = extract_characters_regex(pred)
        model_pred["extracted_model_prediction"] = extracted_pred
        gt = model_pred.get("ground_truth") or model_pred.get("answer", "")
        
        # 获取类型并映射到官方类型名称
        original_type = model_pred.get("type", "")
        mapped_type = TYPE_MAPPING.get(original_type, None)
        
        if extracted_pred != "":
            valid_num += 1
            if extracted_pred == gt:
                correct_num += 1
            
            # 按类型统计
            if mapped_type and mapped_type in type_stats:
                type_stats[mapped_type]["valid_num"] += 1
                if extracted_pred == gt:
                    type_stats[mapped_type]["correct_num"] += 1
        
        total_num += 1
        
        # 按类型统计总数
        if mapped_type and mapped_type in type_stats:
            type_stats[mapped_type]["total_num"] += 1

    accuracy = correct_num / valid_num * 100 if valid_num > 0 else "No valid predictions."
    
    # 计算各类型的准确率
    type_accuracies = {}
    for type_name, stats in type_stats.items():
        if stats["valid_num"] > 0:
            type_accuracies[type_name] = stats["correct_num"] / stats["valid_num"] * 100
        else:
            type_accuracies[type_name] = "No valid predictions."
    
    result = {
        "accuracy": accuracy,
        "total_num": total_num,
        "valid_num": valid_num,
        "correct_num": correct_num,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mc_extract": "think_llm" if think_extract else "regex",
        "type_accuracies": type_accuracies,
        "type_stats": {k: v for k, v in type_stats.items()},
        "model_predictons": model_predictions
    }

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    return result


# ============ 批量评测函数 ============

def bulk_eval_gar_detailed(
    input_dir: str,
    pattern: str,
    log_dir: str,
    summary_out: Optional[str],
    base_url: str,
    model_name: str,
    num_process: int = 8,
    layer_range: Optional[Tuple[int, int]] = None,
):
    """
    批量评测 GAR-detailed 格式的结果文件
    
    Args:
        input_dir: 输入目录
        pattern: 文件匹配模式（glob）
        log_dir: 保存每个文件详细评测结果的目录
        summary_out: 汇总 CSV 文件路径（可选）
        base_url: vLLM 服务地址
        model_name: 评测模型名称
        num_process: 并行处理数量
        layer_range: 层范围过滤（可选，如 (0, 35)）
    
    Returns:
        包含所有文件评测结果的列表
    """
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    assert files, f"No files matched: {os.path.join(input_dir, pattern)}"

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

    os.makedirs(log_dir, exist_ok=True)

    rows = []
    for fp in tqdm(files, desc="Bulk GAR-detailed evaluating (vLLM)"):
        with open(fp, "r", encoding="utf-8") as f:
            preds = json.load(f)

        base = os.path.basename(fp)
        log_path = os.path.join(log_dir, base + ".metrics.json")

        res = compute_metric_gar_detailed(
            preds,
            log_path,
            base_url=base_url,
            model_name=model_name,
            num_process=num_process,
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
            "true_num": res["true_num"],
            "total_num": res["total_num"],
        })

    if summary_out:
        os.makedirs(os.path.dirname(summary_out), exist_ok=True)
        with open(summary_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file", "layer", "multiplier", "behavior", "type", "model_name", "model_size",
                    "accuracy", "valid_num", "true_num", "total_num",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Summary CSV] {summary_out}  ({len(rows)} runs)")
    return rows


def bulk_eval_gar_simple(
    input_dir: str,
    pattern: str,
    log_dir: str,
    summary_out: Optional[str],
    base_url: str,
    model_name: str,
    num_process: int = 8,
    layer_range: Optional[Tuple[int, int]] = None,
):
    """
    批量评测 GAR-simple 格式的结果文件
    
    Args:
        input_dir: 输入目录
        pattern: 文件匹配模式（glob）
        log_dir: 保存每个文件详细评测结果的目录
        summary_out: 汇总 CSV 文件路径（可选）
        base_url: vLLM 服务地址
        model_name: 评测模型名称
        num_process: 并行处理数量
        layer_range: 层范围过滤（可选，如 (0, 35)）
    
    Returns:
        包含所有文件评测结果的列表
    """
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    assert files, f"No files matched: {os.path.join(input_dir, pattern)}"

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

    os.makedirs(log_dir, exist_ok=True)

    rows = []
    for fp in tqdm(files, desc="Bulk GAR-simple evaluating (vLLM)"):
        with open(fp, "r", encoding="utf-8") as f:
            preds = json.load(f)

        base = os.path.basename(fp)
        log_path = os.path.join(log_dir, base + ".metrics.json")

        res = compute_metric_gar_simple(
            preds,
            log_path,
            base_url=base_url,
            model_name=model_name,
            num_process=num_process,
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
            "true_num": res["true_num"],
            "total_num": res["total_num"],
        })

    if summary_out:
        os.makedirs(os.path.dirname(summary_out), exist_ok=True)
        with open(summary_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file", "layer", "multiplier", "behavior", "type", "model_name", "model_size",
                    "accuracy", "valid_num", "true_num", "total_num",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Summary CSV] {summary_out}  ({len(rows)} runs)")
    return rows


def bulk_eval_mc_gar_vqa(
    input_dir: str,
    pattern: str,
    log_dir: str,
    summary_out: Optional[str],
    layer_range: Optional[Tuple[int, int]] = None,
    *,
    think_extract: bool = False,
    base_url: str = "http://127.0.0.1:23333/v1",
    model_name: str = "Qwen/Qwen2.5-72B-Instruct-AWQ",
    num_process: int = 8,
    api_key: str = "EMPTY",
):
    """
    批量评测 GAR VQA 格式的结果文件，支持按类型分类统计
    
    Args:
        input_dir: 输入目录
        pattern: 文件匹配模式（glob）
        log_dir: 保存每个文件详细评测结果的目录
        summary_out: 汇总 CSV 文件路径（可选）
        layer_range: 层范围过滤（可选，如 (0, 35)）
    
    Returns:
        包含所有文件评测结果的列表
    """
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    assert files, f"No files matched: {os.path.join(input_dir, pattern)}"

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

    os.makedirs(log_dir, exist_ok=True)
    rows = []  # for CSV

    for fp in tqdm(files, desc="Bulk GAR VQA MC evaluating"):
        with open(fp, "r", encoding="utf-8") as f:
            preds = json.load(f)

        # log file name
        base = os.path.basename(fp)
        log_path = os.path.join(log_dir, base + ".metrics.json")

        res = compute_metric_mc_gar_vqa(
            preds,
            log_path,
            think_extract=think_extract,
            base_url=base_url,
            model_name=model_name,
            num_process=num_process,
            api_key=api_key,
        )
        meta = parse_meta_from_filename(base)

        # 获取各类型的准确率
        type_accuracies = res.get("type_accuracies", {})
        
        row = {
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
            # 按类型分类的准确率（与图片中官方类型对应）
            "acc_Color": type_accuracies.get("Color", "N/A"),
            "acc_Shape": type_accuracies.get("Shape", "N/A"),
            "acc_Texture": type_accuracies.get("Texture", "N/A"),
            "acc_Material": type_accuracies.get("Material", "N/A"),
            "acc_Position": type_accuracies.get("Position", "N/A"),
            "acc_Non-Entity": type_accuracies.get("Non-Entity", "N/A"),
            "acc_Relation": type_accuracies.get("Relation", "N/A"),
        }
        rows.append(row)

    if summary_out:
        os.makedirs(os.path.dirname(summary_out), exist_ok=True)
        with open(summary_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file", "layer", "multiplier", "behavior", "type", "model_name", "model_size",
                    "accuracy", "valid_num", "correct_num", "total_num",
                    "acc_Color", "acc_Shape", "acc_Texture", "acc_Material",
                    "acc_Position", "acc_Non-Entity", "acc_Relation"
                ]
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"[Summary CSV] {summary_out}  ({len(rows)} runs)")
    return rows


# ============ 主程序入口 ============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified GAR evaluation tool for VQA, Simple, and Detailed formats")
    
    # 评测类型
    parser.add_argument("--eval_type", type=str, choices=["vqa", "simple", "detailed"], required=True,
                       help="Evaluation type: 'vqa' for GAR VQA (multi-choice), 'simple' for GAR-simple, 'detailed' for GAR-detailed")
    
    # 批量评测参数
    parser.add_argument("--input_dir", type=str, required=True, help="directory that contains prediction files")
    parser.add_argument("--pattern", type=str, default=None, help="glob pattern to match files (auto-detected if not provided)")
    parser.add_argument("--log_dir", type=str, required=True, help="directory to save per-file metrics")
    parser.add_argument("--summary_out", type=str, default=None, help="CSV to save summary of all files")
    
    # vLLM 相关参数（用于 simple 和 detailed）
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="vLLM OpenAI API endpoint")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-72B-Instruct-AWQ", help="served model name for vLLM")
    parser.add_argument("--num_process", type=int, default=8, help="number of parallel processes")
    parser.add_argument(
        "--think_mc",
        action="store_true",
        help="for vqa: use LLM to extract A/B/C/D (gar_image_think_mc_qa style outputs)",
    )
    
    # 层范围参数
    parser.add_argument("--layer_range", nargs=2, type=int, default=None, help="only evaluate layers in this range (e.g., 0 35)")

    args = parser.parse_args()

    # 如果没有提供 pattern，根据 eval_type 自动设置
    if args.pattern is None:
        if args.eval_type == "detailed":
            args.pattern = "results_layer_*_multiplier_*_behavior_refer_type_gar_image_detail_oe_qa_model_name_*_model_size_*.json"
        elif args.eval_type == "simple":
            args.pattern = "results_layer_*_multiplier_*_behavior_refer_type_gar_image_simple_oe_qa_model_name_*_model_size_*.json"
        else:  # vqa
            if args.think_mc:
                args.pattern = "results_layer_*_multiplier_*_behavior_refer_type_gar_image_think_mc_qa*"
            else:
                args.pattern = "results_layer_*_multiplier_*_behavior_refer_type_gar_image_mc_qa*"

    # 根据评测类型调用相应的函数
    if args.eval_type == "detailed":
        bulk_eval_gar_detailed(
            input_dir=args.input_dir,
            pattern=args.pattern,
            log_dir=args.log_dir,
            summary_out=args.summary_out,
            base_url=args.base_url,
            model_name=args.model_name,
            num_process=args.num_process,
            layer_range=tuple(args.layer_range) if args.layer_range else None,
        )
    elif args.eval_type == "simple":
        bulk_eval_gar_simple(
            input_dir=args.input_dir,
            pattern=args.pattern,
            log_dir=args.log_dir,
            summary_out=args.summary_out,
            base_url=args.base_url,
            model_name=args.model_name,
            num_process=args.num_process,
            layer_range=tuple(args.layer_range) if args.layer_range else None,
        )
    else:  # vqa
        bulk_eval_mc_gar_vqa(
            input_dir=args.input_dir,
            pattern=args.pattern,
            log_dir=args.log_dir,
            summary_out=args.summary_out,
            layer_range=tuple(args.layer_range) if args.layer_range else None,
            think_extract=args.think_mc,
            base_url=args.base_url,
            model_name=args.model_name,
            num_process=args.num_process,
        )

    print("Done.")
