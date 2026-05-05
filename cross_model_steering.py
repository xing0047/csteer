"""
Cross-model steering (single fixed mapping): Qwen3VL CAA vectors -> InternVL 3.5 inference.

Vectors are loaded from NORMALIZED_VECTOR/.../qwen3vl_<qwen_size>/...
Results are written under RESULT/.../internvl3_5_<internvl_size>/...

Usage:
  python cross_model_steering.py --behaviors refer --layers 10 --multipliers 0.5 1 --type gar_image_mc_qa \\
    --qwen_size 8b --internvl_size 8b --vector_dir inst_it_image_mo_dev --output_dir inst_it_image_mo_dev
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch as t
from tqdm import tqdm

import prompting_with_steering as pws
from behaviors import (
    ALL_BEHAVIORS,
    get_blink_image_mc_data,
    get_cvbench_image_mc_data,
    get_gar_caption_detailed_data,
    get_gar_caption_simple_data,
    get_gar_image_mc_data,
    get_inst_it_image_mc_data,
    get_inst_it_image_oe_data,
    get_inst_it_video_mc_data,
    get_inst_it_video_oe_data,
    get_results_dir,
    get_steering_vector,
    get_vip_bench_data,
)
from steering_settings import SteeringSettings
from utils.mwrapper_map import model_to_wrapper_map

# Fixed mapping for this script only.
VECTOR_MODEL_NAME = "qwen3vl"
INFER_MODEL_NAME = "internvl3_5"


def test_cross_model_steering(
    layers: List[int],
    multipliers: List[float],
    settings: SteeringSettings,
    qwen_size: str,
    internvl_size: str,
    overwrite: bool = False,
    resume: bool = False,
    vector_dir: str = "",
    output_dir: str = "",
    verbose: bool = False,
    use_flash_attn: bool = False,
    in_query: bool = False,
    marker_only: bool = False,
    not_marker: bool = False,
    noref: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load steering vectors from Qwen3VL checkpoints; run generation on InternVL 3.5.
    """
    failures: List[Dict[str, Any]] = []
    effective_output_dir = f"{output_dir}_noref" if noref else output_dir
    save_results_dir = get_results_dir(
        settings.behavior, INFER_MODEL_NAME, internvl_size, effective_output_dir
    )
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)

    print(
        f"[cross_model_steering] vectors: {VECTOR_MODEL_NAME}_{qwen_size} "
        f"-> inference: {INFER_MODEL_NAME}_{internvl_size}",
        flush=True,
    )

    process_methods = {
        "vip_image_oe_qa": pws.process_item_vip_image_oe_qa,
        "blink_image_mc_qa": pws.process_item_blink_image_mc_qa,
        "blink_image_think_mc_qa": pws.process_item_blink_image_mc_qa,
        "cvbench_image_mc_qa": pws.process_item_cvbench_image_mc_qa,
        "cvbench_image_think_mc_qa": pws.process_item_cvbench_image_mc_qa,
        "inst_it_image_mc_qa": pws.process_item_inst_it_image_mc_qa,
        "inst_it_image_think_mc_qa": pws.process_item_inst_it_image_mc_qa,
        "inst_it_image_oe_qa": pws.process_item_inst_it_image_oe_qa,
        "inst_it_video_mc_qa": pws.process_item_inst_it_video_mc_qa,
        "inst_it_video_oe_qa": pws.process_item_inst_it_video_oe_qa,
        "gar_image_mc_qa": pws.process_item_gar_image_mc_qa,
        "gar_image_think_mc_qa": pws.process_item_gar_image_mc_qa,
        "gar_image_simple_oe_qa": pws.process_item_gar_image_simple_oe_qa,
        "gar_image_detail_oe_qa": pws.process_item_gar_image_detail_oe_qa,
    }
    get_test_data = {
        "vip_image_oe_qa": (lambda: get_vip_bench_data(noref=True)) if noref else get_vip_bench_data,
        "blink_image_mc_qa": (lambda: get_blink_image_mc_data(noref=True)) if noref else get_blink_image_mc_data,
        "blink_image_think_mc_qa": (lambda: get_blink_image_mc_data(noref=True)) if noref else get_blink_image_mc_data,
        "cvbench_image_mc_qa": (lambda: get_cvbench_image_mc_data(noref=True)) if noref else get_cvbench_image_mc_data,
        "cvbench_image_think_mc_qa": (lambda: get_cvbench_image_mc_data(noref=True)) if noref else get_cvbench_image_mc_data,
        "inst_it_image_mc_qa": (lambda: get_inst_it_image_mc_data(noref=True)) if noref else get_inst_it_image_mc_data,
        "inst_it_image_think_mc_qa": (lambda: get_inst_it_image_mc_data(noref=True)) if noref else get_inst_it_image_mc_data,
        "inst_it_image_oe_qa": (lambda: get_inst_it_image_oe_data(noref=True)) if noref else get_inst_it_image_oe_data,
        "inst_it_video_mc_qa": (lambda: get_inst_it_video_mc_data(noref=True)) if noref else get_inst_it_video_mc_data,
        "inst_it_video_oe_qa": (lambda: get_inst_it_video_oe_data(noref=True)) if noref else get_inst_it_video_oe_data,
        "gar_image_mc_qa": (lambda: get_gar_image_mc_data(noref=True)) if noref else get_gar_image_mc_data,
        "gar_image_think_mc_qa": (lambda: get_gar_image_mc_data(noref=True)) if noref else get_gar_image_mc_data,
        "gar_image_simple_oe_qa": (lambda: get_gar_caption_simple_data(noref=True)) if noref else get_gar_caption_simple_data,
        "gar_image_detail_oe_qa": (lambda: get_gar_caption_detailed_data(noref=True)) if noref else get_gar_caption_detailed_data,
    }

    wrapper = model_to_wrapper_map[INFER_MODEL_NAME][internvl_size]
    model = wrapper(
        name=settings.model_name,
        size=settings.model_size,
        use_flash_attn=use_flash_attn,
    )
    model.set_save_internal_decodings(False)
    if noref:
        setattr(model, "noref", True)

    test_data = get_test_data[settings.type]()
    if layers is None or multipliers is None:
        raise ValueError("Requires both --layers and --multipliers.")

    if isinstance(layers, list):
        for layer in layers:
            vector = None
            if not noref:
                vector = get_steering_vector(
                    settings.behavior,
                    VECTOR_MODEL_NAME,
                    qwen_size,
                    vector_dir,
                    layer,
                    normalized=True,
                )
                vector = vector.to(model.device)

            for multiplier in multipliers:
                result_save_suffix = settings.make_result_save_suffix(
                    layer=layer, multiplier=multiplier
                )
                save_filename = os.path.join(
                    save_results_dir,
                    f"results_{result_save_suffix}.json",
                )
                effective_resume = bool(resume) and (not overwrite)

                jsonl_filename = save_filename + "l"
                existing_results: list = []
                seen_keys: set = set()
                if effective_resume:
                    existing_results, seen_keys = pws._load_existing_results_for_resume(save_filename)
                    print(
                        f"[resume] {os.path.basename(save_filename)}: "
                        f"{len(seen_keys)} completed samples will be skipped"
                    )

                if os.path.exists(save_filename) and (not overwrite) and (not effective_resume):
                    print("Found existing", save_filename, "- skipping")
                    continue
                results = []
                os.makedirs(os.path.dirname(save_filename), exist_ok=True)
                jsonl_fp = None
                if effective_resume:
                    jl_mode = "a" if os.path.exists(jsonl_filename) else "w"
                    jsonl_fp = open(jsonl_filename, jl_mode, encoding="utf-8")
                try:
                    for item in tqdm(test_data, desc=f"Layer {layer}, multiplier {multiplier}"):
                        resume_key = pws._item_resume_key(item)
                        if effective_resume and resume_key in seen_keys:
                            continue

                        try:
                            model.reset_all()
                            if not noref:
                                model.set_add_activations(layer, multiplier * vector)
                            qid = str(item.get("question_id", ""))
                            video_path = str(item.get("video_path", ""))
                            normalized_video_path = os.path.normpath(video_path).replace("\\", "/")
                            should_skip_to_gt = (
                                noref
                                and settings.type in ("inst_it_video_mc_qa", "inst_it_video_oe_qa")
                                and normalized_video_path == "videos_raw/133"
                            )
                            if should_skip_to_gt:
                                gt = item.get("answer", "")
                                result = {
                                    "question": item.get("question", ""),
                                    "model_output": gt,
                                    "raw_model_output": gt,
                                    "skip_to_gt": True,
                                    "skip_video_path": item.get("video_path", ""),
                                    "question_id": item.get("question_id", None),
                                }
                                print(
                                    f"[skip_by_video_path->gt] type={settings.type} "
                                    f"qid={item.get('question_id', 'NA')} path={video_path}"
                                )
                            elif "think" in settings.type:
                                result = process_methods[settings.type](
                                    item=item,
                                    model=model,
                                    system_prompt=None,
                                    verbose=verbose,
                                    in_query=in_query,
                                    marker_only=marker_only,
                                    not_marker=not_marker,
                                    think=True,
                                )
                            else:
                                result = process_methods[settings.type](
                                    item=item,
                                    model=model,
                                    system_prompt=None,
                                    verbose=verbose,
                                    in_query=in_query,
                                    marker_only=marker_only,
                                    not_marker=not_marker,
                                )
                            for meta_k in (
                                "question_id",
                                "image_id",
                                "video_id",
                                "sample_id",
                                "source_filename",
                                "image_path",
                                "video_path",
                                "type",
                            ):
                                if meta_k in item and item.get(meta_k) is not None:
                                    result.setdefault(meta_k, item[meta_k])
                            result["_resume_key"] = resume_key
                            result["cross_vector_model"] = VECTOR_MODEL_NAME
                            result["cross_vector_size"] = qwen_size
                            if jsonl_fp is not None:
                                jsonl_fp.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                                jsonl_fp.flush()
                            results.append(result)
                            if effective_resume:
                                seen_keys.add(resume_key)
                        except Exception as e:
                            qtxt = str(item.get("question", "") or "")
                            if len(qtxt) > 240:
                                qtxt = qtxt[:240] + "..."
                            fail_rec = {
                                "resume_key": resume_key,
                                "eval_type": settings.type,
                                "behavior": settings.behavior,
                                "layer": layer,
                                "multiplier": multiplier,
                                "error": f"{type(e).__name__}: {e}",
                                "question_id": item.get("question_id"),
                                "video_path": item.get("video_path"),
                                "image_path": item.get("image_path"),
                                "question_preview": qtxt,
                            }
                            failures.append(fail_rec)
                            print(
                                f"[inference_failed] resume_key={resume_key} {fail_rec['error']}",
                                flush=True,
                            )
                            continue
                finally:
                    if jsonl_fp is not None:
                        jsonl_fp.close()
                with open(save_filename, "w") as f:
                    final_results = existing_results + results if effective_resume else results
                    json.dump(final_results, f, indent=4, ensure_ascii=False, default=str)
        return failures
    raise ValueError("Invalid --layers argument. Expected a list of integers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3VL steering vectors -> InternVL 3.5 inference (fixed cross-model setup)."
    )
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--multipliers", nargs="+", type=float)
    parser.add_argument(
        "--noref",
        action="store_true",
        default=False,
        help="no-refer baseline: skip GAR visual ID overlays (requires layers len=1 and multipliers=[0.0])",
    )
    parser.add_argument("--behaviors", type=str, nargs="+", default=ALL_BEHAVIORS)
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=[
            "vip_image_oe_qa",
            "blink_image_mc_qa",
            "blink_image_think_mc_qa",
            "cvbench_image_mc_qa",
            "cvbench_image_think_mc_qa",
            "inst_it_image_mc_qa",
            "inst_it_image_think_mc_qa",
            "inst_it_image_oe_qa",
            "inst_it_video_mc_qa",
            "inst_it_video_oe_qa",
            "gar_image_mc_qa",
            "gar_image_think_mc_qa",
            "gar_image_simple_oe_qa",
            "gar_image_detail_oe_qa",
        ],
    )
    parser.add_argument(
        "--qwen_size",
        type=str,
        choices=["2b", "8b", "32b"],
        default="8b",
        help="model_size used in Qwen3VL vector path (qwen3vl_<qwen_size>)",
    )
    parser.add_argument(
        "--internvl_size",
        type=str,
        choices=["2b", "8b", "38b"],
        default="8b",
        help="InternVL 3.5 checkpoint size for inference",
    )
    parser.add_argument("--vector_dir", type=str, default="inst_it_image_mo_dev")
    parser.add_argument("--output_dir", type=str, default="inst_it_image_mo_dev")
    parser.add_argument("--use_flash_attn", action="store_true", help="enable flash attention")
    parser.add_argument("--system_prompt", type=str, default=None, choices=["pos", "neg"], required=False)
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--override_model_weights_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume from partial outputs: incrementally write results_*.jsonl and skip completed items",
    )
    parser.add_argument("--verbose", action="store_true", help="enable verbose outputs")
    parser.add_argument("--in-query", action="store_true", help="[design] in-query")
    parser.add_argument("--marker-only", action="store_true", help="[design] steer only markers during decoding")
    parser.add_argument(
        "--not-marker",
        action="store_true",
        help="[design] steer all decoded tokens except marker bracket spans; mutually exclusive with --marker-only",
    )

    args = parser.parse_args()
    if args.marker_only and args.not_marker:
        parser.error("--marker-only and --not-marker cannot be used together")
    if args.noref:
        assert args.layers is not None and len(args.layers) == 1, "--noref requires exactly one --layers value"
        assert args.multipliers is not None and len(args.multipliers) == 1, "--noref requires exactly one --multipliers value"
        assert abs(float(args.multipliers[0]) - 0.0) < 1e-6, "--noref requires --multipliers 0.0"

    steering_settings = SteeringSettings()
    steering_settings.type = args.type
    steering_settings.system_prompt = args.system_prompt
    steering_settings.override_vector = args.override_vector
    steering_settings.override_vector_model = VECTOR_MODEL_NAME
    steering_settings.use_base_model = args.use_base_model
    steering_settings.model_name = INFER_MODEL_NAME
    steering_settings.model_size = args.internvl_size
    steering_settings.override_model_weights_path = args.override_model_weights_path

    all_inference_failures: List[Dict[str, Any]] = []
    for behavior in args.behaviors:
        steering_settings.behavior = behavior
        all_inference_failures.extend(
            test_cross_model_steering(
                layers=args.layers,
                multipliers=args.multipliers,
                settings=steering_settings,
                qwen_size=args.qwen_size,
                internvl_size=args.internvl_size,
                overwrite=args.overwrite,
                resume=args.resume,
                vector_dir=args.vector_dir,
                output_dir=args.output_dir,
                verbose=args.verbose,
                use_flash_attn=args.use_flash_attn,
                in_query=args.in_query,
                marker_only=args.marker_only,
                not_marker=args.not_marker,
                noref=args.noref,
            )
        )
    if all_inference_failures:
        print(
            "\n=== Inference failures (not saved; re-run with --resume to retry only missing) ===",
            flush=True,
        )
        for i, rec in enumerate(all_inference_failures, 1):
            print(
                f"{i}. resume_key={rec.get('resume_key')} "
                f"type={rec.get('eval_type')} "
                f"qid={rec.get('question_id')} "
                f"err={rec.get('error')}",
                flush=True,
            )
