"""
BLINK Benchmark Evaluation Script

Supports multiple VLM models: ViP-LLaVA, Qwen3VL, InternVL, etc.

Usage:
    python eval_blink.py --model vipllava --model-path mucai/vip-llava-7b --output results/blink_vipllava.json
    python eval_blink.py --model qwen3vl --model-size 8b --output results/blink_qwen3vl.json
"""

import argparse
import json
import os
import re
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Dict, List, Any

import torch

from behaviors import get_blink_image_mc_data, get_cvbench_image_mc_data


# ============ Model factory ============

def load_model(model_type: str, **kwargs):
    """
    Load model based on model type.
    
    Args:
        model_type: Model type ('vipllava', 'qwen3vl', 'sa2va', 'sa2va_qwen3vl', etc.)
        **kwargs: Model-specific arguments
    
    Returns:
        Model wrapper instance
    """
    model_type = model_type.lower()
    
    if model_type == "vipllava":
        from wrappers.vipllava_wrapper import ViPLLaVA_Wrapper
        return ViPLLaVA_Wrapper(
            model_path=kwargs.get("model_path", "mucai/vip-llava-7b"),
            device=kwargs.get("device"),
            load_8bit=kwargs.get("load_8bit", False),
            load_4bit=kwargs.get("load_4bit", False),
        )
    
    elif model_type == "sa2va":
        from wrappers.sa2va_wrapper import Sa2VA_Wrapper
        return Sa2VA_Wrapper(
            model_path=kwargs.get("model_path", "ByteDance/Sa2VA-Qwen3-VL-4B"),
            device=kwargs.get("device"),
            use_flash_attn=kwargs.get("use_flash_attn", True),
        )
    
    # elif model_type == "sa2va_qwen3vl":
    #     from wrappers.sa2va_wrapper import Sa2VA_Qwen3VL_Wrapper
    #     return Sa2VA_Qwen3VL_Wrapper(
    #         size=kwargs.get("model_size", "4b"),
    #         device=kwargs.get("device"),
    #         use_flash_attn=kwargs.get("use_flash_attn", True),
    #     )
    
    # elif model_type == "sa2va_qwen25vl":
    #     from wrappers.sa2va_wrapper import Sa2VA_Qwen25VL_Wrapper
    #     return Sa2VA_Qwen25VL_Wrapper(
    #         size=kwargs.get("model_size", "7b"),
    #         device=kwargs.get("device"),
    #         use_flash_attn=kwargs.get("use_flash_attn", True),
    #     )
    
    # elif model_type == "sa2va_internvl":
    #     from wrappers.sa2va_wrapper import Sa2VA_InternVL_Wrapper
    #     return Sa2VA_InternVL_Wrapper(
    #         size=kwargs.get("model_size", "8b"),
    #         device=kwargs.get("device"),
    #         use_flash_attn=kwargs.get("use_flash_attn", True),
    #     )
    
    # elif model_type == "qwen3vl":
    #     from wrappers.qwen3vl_wrapper import Qwen3VL_Wrapper
    #     return Qwen3VL_Wrapper(
    #         name="qwen3vl",
    #         size=kwargs.get("model_size", "8b"),
    #         use_flash_attn=kwargs.get("use_flash_attn", False),
    #         device=kwargs.get("device"),
    #     )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# ============ Evaluation ============

def extract_answer(response: str) -> str:
    """
    Extract answer letter (A/B/C/D) from model response.
    
    Args:
        response: Model's raw response string
    
    Returns:
        Extracted answer letter or empty string
    """
    response = response.strip().upper()
    
    answer_prefixes = [
        "THE BEST ANSWER IS",
        "THE CORRECT ANSWER IS",
        "THE ANSWER IS",
        "ANSWER:",
        "ANSWER",
        "OPTION:",
        "OPTION",
    ]
    
    for prefix in answer_prefixes:
        response = response.replace(prefix, "")
    
    match = re.search(r"[ABCD]", response)
    return match.group(0) if match else ""


def evaluate_blink(
    model,
    model_type: str,
    output_file: str,
    task_type: str = "blink_image_mc_qa",
    max_samples: Optional[int] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    verbose: bool = False,
    subsets: Optional[List[str]] = None,
):
    """
    Run BLINK benchmark evaluation.
    
    Args:
        model: Model wrapper instance
        model_type: Model type string
        output_file: Path to save results
        task_type: Task type for prompt selection
        max_samples: Maximum number of samples to evaluate (None for all)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        verbose: Whether to print verbose output
        subsets: List of specific subsets to evaluate (None for all)
    """
    print(f"\n{'='*60}")
    print(f"BLINK Benchmark Evaluation")
    print(f"Model: {model_type}")
    print(f"Task: {task_type}")
    print(f"{'='*60}\n")
    
    print("Loading BLINK data...")
    data = get_blink_image_mc_data()
    
    if subsets:
        data = [item for item in data if item["type"] in subsets]
        print(f"Filtered to subsets: {subsets}")
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"Total samples: {len(data)}")
    
    results = []
    
    for item in tqdm(data, desc="Evaluating"):
        try:
            if model_type in ["vipllava", "sa2va", "sa2va_qwen3vl", "sa2va_qwen25vl", "sa2va_internvl"]:
                response = model.generate(
                    dataloader_item=item,
                    task_type=task_type,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    verbose=verbose,
                )
            else:
                images, prompt = model.get_inputs(task_type, item)
                if isinstance(images, list):
                    # Multi-image items: use first frame only in this script
                    image = images[0] if images else None
                else:
                    image = images
                
                response = model.generate_text(
                    image_or_video_path=image,
                    vision_type="image",
                    question=prompt,
                    max_new_tokens=max_new_tokens,
                )
            
            prediction = extract_answer(response)
            
            result = {
                "question_id": item["question_id"],
                "image_id": item.get("image_id", ""),
                "type": item["type"],
                "question": item["question"],
                "choices": item["choices"],
                "ground_truth": item["answer"],
                "model_prediction": response,
                "extracted_prediction": prediction,
                "split": "image",
            }
            
            results.append(result)
            
            if verbose:
                print(f"\n[Q{item['question_id']}] GT: {item['answer']}, Pred: {prediction}")
        
        except Exception as e:
            print(f"\nError processing item {item.get('question_id', 'unknown')}: {e}")
            results.append({
                "question_id": item.get("question_id", "unknown"),
                "type": item.get("type", "unknown"),
                "error": str(e),
                "ground_truth": item.get("answer", ""),
                "model_prediction": "",
                "extracted_prediction": "",
            })
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")
    
    metrics = compute_metrics(results)
    
    metrics_file = output_file.replace(".json", "_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {metrics_file}")
    
    return results, metrics


def evaluate_cvbench(
    model,
    model_type: str,
    output_file: str,
    task_type: str = "cvbench_image_mc_qa",
    max_samples: Optional[int] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    verbose: bool = False,
):
    """
    Run CV-Bench evaluation.
    
    Similar to BLINK but for CV-Bench dataset.
    """
    print(f"\n{'='*60}")
    print(f"CV-Bench Evaluation")
    print(f"Model: {model_type}")
    print(f"Task: {task_type}")
    print(f"{'='*60}\n")
    
    print("Loading CV-Bench data...")
    data = get_cvbench_image_mc_data()
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"Total samples: {len(data)}")
    
    results = []
    
    for item in tqdm(data, desc="Evaluating"):
        try:
            if model_type in ["vipllava", "sa2va", "sa2va_qwen3vl", "sa2va_qwen25vl", "sa2va_internvl"]:
                response = model.generate(
                    dataloader_item=item,
                    task_type=task_type,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    verbose=verbose,
                )
            else:
                images, prompt = model.get_inputs(task_type, item)
                image = images[0] if isinstance(images, list) else images
                response = model.generate_text(
                    image_or_video_path=image,
                    vision_type="image",
                    question=prompt,
                    max_new_tokens=max_new_tokens,
                )
            
            prediction = extract_answer(response)
            
            result = {
                "question_id": item["question_id"],
                "type": item["type"],
                "question": item["question"],
                "ground_truth": item["answer"],
                "model_prediction": response,
                "extracted_prediction": prediction,
                "split": "image",
            }
            results.append(result)
            
        except Exception as e:
            print(f"\nError: {e}")
            results.append({
                "question_id": item.get("question_id", "unknown"),
                "error": str(e),
                "ground_truth": item.get("answer", ""),
                "model_prediction": "",
                "extracted_prediction": "",
            })
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    metrics = compute_metrics(results)
    metrics_file = output_file.replace(".json", "_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Metrics saved to: {metrics_file}")
    
    return results, metrics


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute evaluation metrics from results.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Dictionary containing metrics
    """
    total = 0
    correct = 0
    valid = 0
    
    type_stats = {}
    
    for item in results:
        if "error" in item:
            total += 1
            continue
        
        pred = item.get("extracted_prediction", "")
        gt = item.get("ground_truth", "").strip().upper()
        task_type = item.get("type", "unknown")
        
        if task_type not in type_stats:
            type_stats[task_type] = {"total": 0, "valid": 0, "correct": 0}
        
        type_stats[task_type]["total"] += 1
        total += 1
        
        if pred:
            valid += 1
            type_stats[task_type]["valid"] += 1
            
            if pred == gt:
                correct += 1
                type_stats[task_type]["correct"] += 1
    
    overall_acc = (correct / valid * 100) if valid > 0 else 0.0
    
    per_type_acc = {}
    for t, stats in type_stats.items():
        if stats["valid"] > 0:
            acc = stats["correct"] / stats["valid"] * 100
        else:
            acc = 0.0
        per_type_acc[t] = {
            "accuracy": acc,
            "correct": stats["correct"],
            "valid": stats["valid"],
            "total": stats["total"],
        }
    
    metrics = {
        "overall": {
            "accuracy": overall_acc,
            "correct": correct,
            "valid": valid,
            "total": total,
        },
        "per_type": per_type_acc,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {correct}/{valid} = {overall_acc:.2f}%")
    print(f"\nPer-Task Accuracy:")
    
    BLINK_ABBREV = {
        "Visual_Correspondence": "VCO",
        "Relative_Reflectance": "RRF",
        "Relative_Depth": "RDP",
        "Object_Localization": "OLC",
        "Semantic_Correspondence": "SCO",
        "Functional_Correspondence": "FCO",
        "Relation": "REL",
        "Depth": "DEP",
        "Distance": "DIS",
    }
    
    for t in sorted(per_type_acc.keys()):
        stats = per_type_acc[t]
        abbrev = BLINK_ABBREV.get(t, t[:3].upper())
        print(f"  {abbrev} ({t}): {stats['correct']}/{stats['valid']} = {stats['accuracy']:.2f}%")
    
    print(f"{'='*60}\n")
    
    return metrics


# ============ CLI ============

def main():
    parser = argparse.ArgumentParser(description="BLINK Benchmark Evaluation")
    
    parser.add_argument(
        "--model",
        type=str,
        default="vipllava",
        choices=["vipllava", "sa2va", "sa2va_qwen3vl", "sa2va_qwen25vl", "sa2va_internvl"],
        help="Model type to use",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="mucai/vip-llava-7b",
        help="Path to model (for vipllava)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="8b",
        choices=["8b"],
        help="Model size (for qwen3vl / internvl backends in Sa2VA)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., cuda:0)",
    )
    parser.add_argument(
        "--load-8bit",
        action="store_true",
        help="Load model in 8-bit precision",
    )
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="Load model in 4-bit precision",
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable flash attention (for Sa2VA models)",
    )
    
    parser.add_argument(
        "--benchmark",
        type=str,
        default="blink",
        choices=["blink", "cvbench"],
        help="Benchmark to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/blink_results.json",
        help="Output file path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=None,
        help="Specific subsets to evaluate (e.g., Visual_Correspondence Relative_Depth)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    args = parser.parse_args()

    if args.benchmark == "blink":
        task_type = "blink_image_mc_qa"
    else:
        task_type = "cvbench_image_mc_qa"
    
    print(f"Loading model: {args.model}")
    model = load_model(
        model_type=args.model,
        model_path=args.model_path,
        model_size=args.model_size,
        device=args.device,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        use_flash_attn=not args.no_flash_attn,
    )
    
    if args.benchmark == "blink":
        results, metrics = evaluate_blink(
            model=model,
            model_type=args.model,
            output_file=args.output,
            task_type=task_type,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            verbose=args.verbose,
            subsets=args.subsets,
        )
    else:
        results, metrics = evaluate_cvbench(
            model=model,
            model_type=args.model,
            output_file=args.output,
            task_type=task_type,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            verbose=args.verbose,
        )
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
