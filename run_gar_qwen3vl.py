#!/usr/bin/env python3
"""
GAR (MC / Simple / Detailed) inference with Qwen3-VL: greedy decoding only.
Outputs JSON lists compatible with judge/evaluate_gar.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from behaviors import (  # noqa: E402
    get_gar_caption_detailed_data,
    get_gar_caption_simple_data,
    get_gar_image_mc_data,
)
from utils.cd_qwen3vl_wrapper import (  # noqa: E402
    ADD_FROM_POS_CHAT,
    Qwen3VL_Wrapper,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _strip_assistant(raw: str) -> str:
    return raw.split(ADD_FROM_POS_CHAT)[-1].strip()


def run_mc(
    model: Qwen3VL_Wrapper,
    *,
    max_num: int | None,
    think: bool,
    gen_kwargs: dict,
    verbose: bool = False,
) -> list:
    data = get_gar_image_mc_data(max_num=max_num)
    results = []
    for item in tqdm(data, desc="gar_mc"):
        model.reset_all()
        if think:
            image, question = model.get_inputs("gar_image_think_mc_qa", item)
        else:
            image, question = model.get_inputs("gar_image_mc_qa", item)
        raw = model.generate_text(
            image,
            vision_type="image",
            question=question,
            verbose=verbose,
            **gen_kwargs,
        )
        if verbose:
            print(f"[a]: \033[93m{raw}\033[0m")
        results.append(
            {
                "image_path": item["image_path"],
                "mask_rles": item["mask_rles"],
                "question": question,
                "type": item["type"],
                "answer": item["answer"],
                "ground_truth": item["answer"],
                "model_output": _strip_assistant(raw),
            }
        )
    return results


def run_simple(
    model: Qwen3VL_Wrapper,
    *,
    max_num: int | None,
    gen_kwargs: dict,
    verbose: bool = False,
) -> list:
    data = get_gar_caption_simple_data(max_num=max_num)
    results = []
    for item in tqdm(data, desc="gar_simple"):
        model.reset_all()
        image, question = model.get_inputs("gar_image_simple_oe_qa", item)
        raw = model.generate_text(
            image,
            vision_type="image",
            question=question,
            verbose=verbose,
            **gen_kwargs,
        )
        if verbose:
            print(f"[a]: \033[93m{raw}\033[0m")
        results.append(
            {
                "image_path": item["image_path"],
                "mask_rles": item["mask_rles"],
                "question": item["question"],
                "answer": item["answer"],
                "model_output": _strip_assistant(raw),
            }
        )
    return results


def run_detailed(
    model: Qwen3VL_Wrapper,
    *,
    max_num: int | None,
    gen_kwargs: dict,
    verbose: bool = False,
) -> list:
    data = get_gar_caption_detailed_data(max_num=max_num)
    results = []
    for item in tqdm(data, desc="gar_detailed"):
        model.reset_all()
        image, question = model.get_inputs("gar_image_detail_oe_qa", item)
        raw = model.generate_text(
            image,
            vision_type="image",
            question=question,
            verbose=verbose,
            **gen_kwargs,
        )
        if verbose:
            print(f"[a]: \033[93m{raw}\033[0m")
        results.append(
            {
                "image": item["image_path"],
                "subject_name": item.get("subject_name", ""),
                "object_name": item.get("object_name", ""),
                "predicate_name": item.get("predicate_name", ""),
                "question": item["question"],
                "mask_rles": item["mask_rles"],
                "model_output": _strip_assistant(raw),
                "split": "gar_image_detail_oe_qa",
            }
        )
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["mc", "simple", "detailed"], required=True)
    p.add_argument("--out", type=str, default="", help="Output JSON path")
    p.add_argument("--model_size", type=str, default="8b")
    p.add_argument("--use_flash_attn", action="store_true")
    p.add_argument("--max_num", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Greedy decoding uses 0; passed to generate (no effect when do_sample=False)",
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Unused when do_sample=False; kept for CLI compatibility",
    )
    p.add_argument("--think", action="store_true", help="GAR-MC with think prompt template")
    p.add_argument("--verbose", action="store_true", help="Print question [q] and full raw answer [a] per sample (like prompting_with_steering.py)")
    args = p.parse_args()

    set_seed(args.seed)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    model = Qwen3VL_Wrapper(name="qwen3vl", size=args.model_size, use_flash_attn=args.use_flash_attn)
    model.set_save_internal_decodings(False)

    if args.task == "mc":
        results = run_mc(
            model,
            max_num=args.max_num,
            think=args.think,
            gen_kwargs=gen_kwargs,
            verbose=args.verbose,
        )
        suffix = "think_mc" if args.think else "mc"
    elif args.task == "simple":
        results = run_simple(
            model,
            max_num=args.max_num,
            gen_kwargs=gen_kwargs,
            verbose=args.verbose,
        )
        suffix = "simple"
    else:
        results = run_detailed(
            model,
            max_num=args.max_num,
            gen_kwargs=gen_kwargs,
            verbose=args.verbose,
        )
        suffix = "detailed"

    out = args.out or os.path.join(
        BASE_DIR,
        "..",
        "RESULT",
        "gar_qwen3vl",
        args.model_size,
        f"results_gar_{suffix}.json",
    )
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} items to {out}")


if __name__ == "__main__":
    main()
