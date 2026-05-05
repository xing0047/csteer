"""
Use CAA to steer the model

Usage:
python prompting_with_steering.py --behaviors refer --layers 10 --multipliers 0.1 0.5 1 2 5 10 --type ab --use_base_model --model_size 8b
"""

import json
import hashlib
import os
import argparse
import torch as t

from typing import List, Dict, Optional, Any
from tqdm import tqdm
from steering_settings import SteeringSettings
from behaviors import (
    get_steering_vector,
    get_vip_bench_data,
    get_dlc_bench_data,
    get_blink_image_mc_data,
    get_cvbench_image_mc_data,
    get_inst_it_image_mc_data,
    get_inst_it_image_oe_data,
    get_inst_it_video_mc_data,
    get_inst_it_video_oe_data,
    get_gar_image_mc_data,
    get_gar_caption_simple_data,
    get_gar_caption_detailed_data,
    get_results_dir,
    ALL_BEHAVIORS,
)
from utils.helpers import get_model_to_wrapper_map
from utils.tokenize import (
    ADD_FROM_POS_CHAT, 
)

API_MODELS = {"gpt-4o", "gemini-2.5-pro", "o3"}


def _norm_field(v: Any) -> str:
    return str(v if v is not None else "").strip()


def _resume_key_from_dict(d: Dict[str, Any]) -> str:
    """
    Stable key for a dataloader item or a saved result row.
    Must not depend on PIL tensors or other non-stable objects.
    """
    for k in ("question_id", "sample_id", "idx", "image_id", "video_id", "source_filename"):
        v = d.get(k, None)
        if v is not None and str(v) != "":
            return f"{k}:{v}"
    q = _norm_field(d.get("question"))
    ip = _norm_field(d.get("image_path"))
    vp = _norm_field(d.get("video_path"))
    if q and (ip or vp):
        h = hashlib.sha256(f"{q}\n{ip}\n{vp}".encode("utf-8")).hexdigest()
        return f"qh:{h}"
    if q:
        h = hashlib.sha256(q.encode("utf-8")).hexdigest()
        return f"qonly:{h}"
    stable = {k: d[k] for k in ("type",) if k in d and d[k] is not None}
    if stable:
        try:
            return "meta:" + json.dumps(stable, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            pass
    try:
        return "json:" + json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return "repr:" + repr(d)


def _item_resume_key(item: Dict[str, Any]) -> str:
    return _resume_key_from_dict(item)


def _seen_key_from_saved_row(obj: Dict[str, Any]) -> str:
    if not isinstance(obj, dict):
        return ""
    k = obj.get("_resume_key")
    if k is not None and str(k) != "":
        return str(k)
    return _resume_key_from_dict(obj)


def _load_existing_results_for_resume(save_filename: str) -> tuple[list, set]:
    """
    Return (existing_results, seen_keys) from either jsonl (preferred) or json array.
    Rows without _resume_key still contribute to seen_keys via _resume_key_from_dict.
    """
    existing_results: list = []
    seen: set = set()
    jsonl_path = save_filename + "l"  # results_*.jsonl

    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                existing_results.append(obj)
                sk = _seen_key_from_saved_row(obj)
                if sk:
                    seen.add(sk)
        return existing_results, seen

    if os.path.exists(save_filename):
        try:
            with open(save_filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                existing_results = data
                for obj in existing_results:
                    if isinstance(obj, dict):
                        sk = _seen_key_from_saved_row(obj)
                        if sk:
                            seen.add(sk)
        except Exception:
            pass

    return existing_results, seen

def process_item_vip_image_oe_qa(
    item: Dict[str, str],
    model: Any,
    system_prompt: Optional[str],
    verbose: bool = False,
    in_query: bool = False,
    marker_only: bool = False,
    not_marker: bool = False,
) -> Dict[str, str]:
    image, question = model.get_inputs("vip_image_oe_qa", item)
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    model_output = model.generate_text(
        image, vision_type="image", question=question, in_query=in_query, 
        marker_only=marker_only, not_marker=not_marker, verbose=verbose, system_prompt=system_prompt, **generation_config
    )
    if verbose:
        print(f"[a]: \033[93m{model_output}\033[0m")
    return {
        "question": question,
        "model_output": model_output.split(ADD_FROM_POS_CHAT)[-1].strip(),
        "raw_model_output": model_output,
        "answer": item["answer"],
        "capability": item["capability"]
    }

def process_item_dlc_bench_qa(
    item: Dict[str, str],
    model: Any,
    system_prompt: Optional[str],
) -> Dict[str, str]:
    raise NotImplementedError
    pixel_values, question = model.get_inputs("dlc_bench_qa", item, box=...).to(t.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    model_output, question = model.generate_text(
        pixel_values, question=question, system_prompt=system_prompt, **generation_config
    )
    print(f"\033[93m{model_output}\033[0m")
    return {
        "question": question,
        "model_output": model_output.split(ADD_FROM_POS_CHAT)[-1].strip(),
        "raw_model_output": model_output,
    }

def process_item_blink_image_mc_qa(
    item: Dict[str, Any],
    model,
    system_prompt: Optional[str],
    verbose: bool = False,
    in_query: bool = False,
    marker_only: bool = False,
    not_marker: bool = False,
) -> Dict[str, Any]:
    images, question = model.get_inputs("blink_image_mc_qa", item)
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    if len(images) == 1:
        model_output = model.generate_text(
            images[0], vision_type="image", question=question, in_query=in_query, 
            marker_only=marker_only, not_marker=not_marker, verbose=verbose, system_prompt=system_prompt, **generation_config
        )
    elif len(images) > 1:
        model_output = model.generate_text(
            images, vision_type="multi_image", question=question, in_query=in_query, 
            marker_only=marker_only, not_marker=not_marker, verbose=verbose, system_prompt=system_prompt, **generation_config
        )
    else:
        raise NotImplementedError
    if verbose:
        print(f"[a]: \033[93m{model_output}\033[0m")

    return {
        "image_path": None,
        "type": item["type"],
        "question": item["question"],
        "answer": item["answer"],
        "model_output": model_output.split(ADD_FROM_POS_CHAT)[-1].strip(),
    }

def process_item_cvbench_image_mc_qa(
    item: Dict[str, Any],
    model,
    system_prompt: Optional[str],
    verbose: bool = False,
    in_query: bool = False,
    marker_only: bool = False,
    not_marker: bool = False,
) -> Dict[str, Any]:
    image, question = model.get_inputs("cvbench_image_mc_qa", item)
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    model_output = model.generate_text(
        image, vision_type="image", question=question, in_query=in_query, 
        marker_only=marker_only, not_marker=not_marker, verbose=verbose, system_prompt=system_prompt, **generation_config
    )
    if verbose:
        print(f"[a]: \033[93m{model_output}\033[0m")

    return {
        "image_path": None,
        "type": item["type"],
        "source_filename": item["source_filename"],
        "question": item["question"],
        "answer": item["answer"],
        "model_output": model_output.split(ADD_FROM_POS_CHAT)[-1].strip(),
    }

def process_item_inst_it_image_mc_qa(
    item: Dict[str, str],
    model: Any,
    system_prompt: Optional[str],
    verbose=False,
    in_query: bool = False,
    marker_only: bool = False,
    not_marker: bool = False,
) -> Dict[str, str]:
    image, question = model.get_inputs('inst_it_image_mc_qa', item)
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    model_output = model.generate_text(
        image, vision_type="image", question=question, in_query=in_query, 
        marker_only=marker_only, not_marker=not_marker, verbose=verbose, system_prompt=system_prompt, **generation_config
    )
    if verbose:
        print(f"[a]: \033[93m{model_output}\033[0m")
    return {
        "question": question,
        "model_output": model_output.split(ADD_FROM_POS_CHAT)[-1].strip(),
        "raw_model_output": model_output,
    }

def process_item_inst_it_image_oe_qa(
    item: Dict[str, str],
    model: Any,
    system_prompt: Optional[str],
    in_query: bool = False,
    marker_only: bool = False,
    not_marker: bool = False,
    verbose=False,
) -> Dict[str, str]:
    image, question = model.get_inputs('inst_it_image_oe_qa', item)
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    model_output = model.generate_text(
        image, vision_type="image", question=question, in_query=in_query, 
        marker_only=marker_only, not_marker=not_marker, verbose=verbose, system_prompt=system_prompt, **generation_config
    )
    if verbose:
        print(f"[a]: \033[93m{model_output}\033[0m")
    return {
        "question": question,
        "model_output": model_output.split(ADD_FROM_POS_CHAT)[-1].strip(),
        "raw_model_output": model_output,
    }

def process_item_inst_it_video_mc_qa(
    item: Dict[str, str],
    model: Any,
    system_prompt: Optional[str],
    verbose=False,
    in_query: bool = False,
    marker_only: bool = False,
    not_marker: bool = False,
) -> Dict[str, str]:
    video_path, question = model.get_inputs('inst_it_video_mc_qa', item)
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    model_output = model.generate_text(
        video_path, vision_type="video", question=question, in_query=in_query, 
        marker_only=marker_only, not_marker=not_marker, verbose=verbose, system_prompt=system_prompt, **generation_config
    )
    if verbose:
        print(f"[a]: \033[93m{model_output}\033[0m")
    return {
        "question": question,
        "model_output": model_output.split(ADD_FROM_POS_CHAT)[-1].strip(),
        "raw_model_output": model_output,
    }

def process_item_inst_it_video_oe_qa(
    item: Dict[str, str],
    model: Any,
    system_prompt: Optional[str],
    verbose=False,
    in_query: bool = False,
    marker_only: bool = False,
    not_marker: bool = False,
) -> Dict[str, str]:
    video_path, question = model.get_inputs('inst_it_video_oe_qa', item)
    generation_config = dict(max_new_tokens=2048, do_sample=False)
    model_output = model.generate_text(
        video_path, vision_type="video", question=question, in_query=in_query, 
        marker_only=marker_only, not_marker=not_marker, verbose=verbose, system_prompt=system_prompt, **generation_config
    )
    if verbose:
        print(f"[a]: \033[93m{model_output}\033[0m")
    return {
        "question": question,
        "model_output": model_output.split(ADD_FROM_POS_CHAT)[-1].strip(),
        "raw_model_output": model_output,
    }

def process_item_gar_image_mc_qa(
    item: Dict[str, Any],
    model,
    system_prompt: Optional[str],
    verbose: bool = False,
    in_query: bool = False,
    marker_only: bool = False,
    not_marker: bool = False,
) -> Dict[str, Any]:
    image, question = model.get_inputs("gar_image_mc_qa", item)
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    model_output = model.generate_text(
        image, vision_type="image", question=question, in_query=in_query, 
        marker_only=marker_only, not_marker=not_marker, verbose=verbose, system_prompt=system_prompt, **generation_config
    )
    if verbose:
        print(f"[a]: \033[93m{model_output}\033[0m")
    return {
        "image_path": item["image_path"],
        "mask_rles": item["mask_rles"],
        "question": question,
        "type": item["type"],
        "answer": item["answer"],
        "model_output": model_output.split(ADD_FROM_POS_CHAT)[-1].strip(),
    }

def process_item_gar_image_simple_oe_qa(
    item: Dict[str, Any],
    model,
    system_prompt: Optional[str],
    verbose: bool = False,
    in_query: bool = False,
    marker_only: bool = False,
    not_marker: bool = False,
) -> Dict[str, Any]:
    image, question = model.get_inputs("gar_image_simple_oe_qa", item)
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    model_output = model.generate_text(
        image, vision_type="image", question=question, in_query=in_query, 
        marker_only=marker_only, not_marker=not_marker, verbose=verbose, system_prompt=system_prompt, **generation_config
    )
    if verbose:
        print(f"[a]: \033[93m{model_output}\033[0m")

    return {
        "image_path": item["image_path"],
        "mask_rles": item["mask_rles"],
        "question": item["question"],
        "answer": item["answer"],
        "model_output": model_output.split(ADD_FROM_POS_CHAT)[-1].strip(),
    }

def process_item_gar_image_detail_oe_qa(
    item: Dict[str, Any],
    model,
    system_prompt: Optional[str],
    verbose: bool = False,
    in_query: bool = False,
    marker_only: bool = False,
    not_marker: bool = False,
) -> Dict[str, Any]:
    image, question = model.get_inputs('gar_image_detail_oe_qa', item)

    generation_config = dict(max_new_tokens=1024, do_sample=False)
    model_output = model.generate_text(
        image, vision_type="image", question=question, in_query=in_query, 
        marker_only=marker_only, not_marker=not_marker, system_prompt=system_prompt, verbose=verbose, **generation_config
    )
    if verbose:
        print(f"[a]: \033[93m{model_output}\033[0m")

    return {
        "image": item["image_path"],
        "subject_name": item.get("subject_name", ""),
        "object_name": item.get("object_name", ""),
        "predicate_name": item.get("predicate_name", ""),
        "question": item["question"],
        "mask_rles": item["mask_rles"],
        "model_output": model_output.split(ADD_FROM_POS_CHAT)[-1].strip(),
        "split": "gar_image_detail_oe_qa",
    }

def test_steering(
    layers: List[int], 
    multipliers: List[int], 
    settings: SteeringSettings, 
    overwrite: bool = False, 
    resume: bool = False,
    model_name: str = 'internvl3_5', 
    model_size: str = '8b',
    vector_dir: str = '',
    output_dir: str = '',
    verbose: bool = False,
    use_flash_attn: bool = False,
    in_query: bool = False,
    marker_only: bool = False,
    not_marker: bool = False,
    noref: bool = False,
) -> List[Dict[str, Any]]:
    """
    layers: List of layers to test steering on.
    multipliers: List of multipliers to test steering with.
    settings: SteeringSettings object.
    Returns a list of per-sample inference failures (not written to json/jsonl; safe to retry with --resume).
    """
    failures: List[Dict[str, Any]] = []
    model_to_wrapper_map = get_model_to_wrapper_map()
    is_api_model = model_name in API_MODELS
    # Keep JSON filename schema unchanged (judge scripts rely on it),
    # but separate baseline/no-refer results by output directory.
    effective_output_dir = f"{output_dir}_noref" if noref else output_dir
    save_results_dir = get_results_dir(settings.behavior, model_name, model_size, effective_output_dir)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    process_methods = {
        "vip_image_oe_qa": process_item_vip_image_oe_qa,
        # "dlc_bench_qa": process_item_dlc_bench_qa,
        "blink_image_mc_qa": process_item_blink_image_mc_qa,
        "cvbench_image_mc_qa": process_item_cvbench_image_mc_qa,
        "inst_it_image_mc_qa": process_item_inst_it_image_mc_qa,
        "inst_it_image_oe_qa": process_item_inst_it_image_oe_qa,
        "inst_it_video_mc_qa": process_item_inst_it_video_mc_qa,
        "inst_it_video_oe_qa": process_item_inst_it_video_oe_qa,
        "gar_image_mc_qa": process_item_gar_image_mc_qa,
        "gar_image_simple_oe_qa": process_item_gar_image_simple_oe_qa,
        "gar_image_detail_oe_qa": process_item_gar_image_detail_oe_qa,
    }
    get_test_data = {
        # Keep per-dataset defaults unless user explicitly enables --noref baseline.
        "vip_image_oe_qa": (lambda: get_vip_bench_data(noref=True)) if noref else get_vip_bench_data,
        "dlc_bench_qa": get_dlc_bench_data,
        "blink_image_mc_qa": (lambda: get_blink_image_mc_data(noref=True)) if noref else get_blink_image_mc_data,
        "cvbench_image_mc_qa": (lambda: get_cvbench_image_mc_data(noref=True)) if noref else get_cvbench_image_mc_data,
        "inst_it_image_mc_qa": (lambda: get_inst_it_image_mc_data(noref=True)) if noref else get_inst_it_image_mc_data,
        "inst_it_image_oe_qa": (lambda: get_inst_it_image_oe_data(noref=True)) if noref else get_inst_it_image_oe_data,
        "inst_it_video_mc_qa": (lambda: get_inst_it_video_mc_data(noref=True)) if noref else get_inst_it_video_mc_data,
        "inst_it_video_oe_qa": (lambda: get_inst_it_video_oe_data(noref=True)) if noref else get_inst_it_video_oe_data,
        "gar_image_mc_qa": (lambda: get_gar_image_mc_data(noref=True)) if noref else get_gar_image_mc_data,
        "gar_image_simple_oe_qa": (lambda: get_gar_caption_simple_data(noref=True)) if noref else get_gar_caption_simple_data,
        "gar_image_detail_oe_qa": (lambda: get_gar_caption_detailed_data(noref=True)) if noref else get_gar_caption_detailed_data,
    }
    if is_api_model:
        wrapper = model_to_wrapper_map[model_name]
        model = wrapper(
            name=settings.model_name,
        )
    else:
        wrapper = model_to_wrapper_map[model_name][model_size]
        model = wrapper(
            name=settings.model_name,
            size=settings.model_size,
            use_flash_attn=use_flash_attn,
        )
        model.set_save_internal_decodings(False)
    if noref:
        # Signals wrappers to skip GAR visual ID overlays.
        setattr(model, "noref", True)

    test_data = get_test_data[settings.type]()
    # Closed-source API models do not support activation steering in this repo.
    if is_api_model:
        layers = [0]
        multipliers = [0.0]
        noref = True
    elif layers is None or multipliers is None:
        raise ValueError("Open-source models require both --layers and --multipliers.")

    if isinstance(layers, list):
        for layer in layers:
            vector = None
            if (not noref) and (not is_api_model):
                vector = get_steering_vector(
                    settings.behavior, model_name, model_size, vector_dir, layer, normalized=True
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
                # Do not mutate outer `resume` (would break later layers / multipliers).
                effective_resume = bool(resume) and (not overwrite)

                jsonl_filename = save_filename + "l"  # results_*.jsonl
                existing_results: list = []
                seen_keys: set = set()
                if effective_resume:
                    existing_results, seen_keys = _load_existing_results_for_resume(save_filename)
                    print(
                        f"[resume] {os.path.basename(save_filename)}: "
                        f"{len(seen_keys)} completed samples will be skipped"
                    )

                if os.path.exists(save_filename) and (not overwrite) and (not effective_resume):
                    print("Found existing", save_filename, "- skipping")
                    continue
                results = []
                os.makedirs(os.path.dirname(save_filename), exist_ok=True)
                # Only touch jsonl when --resume: avoids truncating a partial .jsonl on a non-resume re-run.
                jsonl_fp = None
                if effective_resume:
                    jl_mode = "a" if os.path.exists(jsonl_filename) else "w"
                    jsonl_fp = open(jsonl_filename, jl_mode, encoding="utf-8")
                try:
                    for item in tqdm(test_data, desc=f"Layer {layer}, multiplier {multiplier}"):
                        resume_key = _item_resume_key(item)
                        if effective_resume and resume_key in seen_keys:
                            continue

                        try:
                            model.reset_all()
                            if (not noref) and (not is_api_model):
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
                            else:
                                result = process_methods[settings.type](
                                    item=item,
                                    model=model,
                                    system_prompt=None,
                                    verbose=verbose,
                                    in_query=in_query,
                                    marker_only=marker_only, not_marker=not_marker,
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
                with open(
                    save_filename,
                    "w",
                ) as f:
                    final_results = existing_results + results if effective_resume else results
                    json.dump(final_results, f, indent=4, ensure_ascii=False, default=str)
        return failures
    else:
        raise ValueError("Invalid --layers argument. Expected a list of integers.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--multipliers", nargs="+", type=float)
    parser.add_argument(
        "--noref",
        action="store_true",
        default=False,
        help="no-refer baseline: skip GAR visual ID overlays (requires layers len=1 and multipliers=[0.0])",
    )
    parser.add_argument("--behaviors",
        type=str,
        nargs="+",
        default=ALL_BEHAVIORS
    )
    parser.add_argument("--type",
        type=str,
        required=True,
        choices=[
            "vip_image_oe_qa",
            "blink_image_mc_qa",
            "cvbench_image_mc_qa",
            "inst_it_image_mc_qa",
            "inst_it_image_oe_qa",
            "inst_it_video_mc_qa",
            "inst_it_video_oe_qa",
            "gar_image_mc_qa",
            "gar_image_simple_oe_qa",
            "gar_image_detail_oe_qa",
        ],
    )
    parser.add_argument("--model_name", type=str, choices=["internvl3_5", "qwen3vl", "gpt-4o", "gemini-2.5-pro", "o3"], required=True)
    parser.add_argument("--model_size", type=str, default="8b", choices=["8b"])
    parser.add_argument("--vector_dir", type=str, default="inst_it_image_mo_dev")
    parser.add_argument("--output_dir", type=str, default="inst_it_image_mo_dev")
    parser.add_argument('--use_flash_attn', action='store_true', help="enable flash attention")
    parser.add_argument("--system_prompt", type=str, default=None, choices=["pos", "neg"], required=False)
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_vector_model", type=str, default=None)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--override_model_weights_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume from partial outputs: incrementally write results_*.jsonl and skip completed items",
    )
    parser.add_argument('--verbose', action='store_true', help="enable verbose outputs")
    parser.add_argument('--in-query', action='store_true', help="[design] in-query")
    parser.add_argument('--marker-only', action='store_true', help="[design] steer only markers during decoding")
    parser.add_argument(
        '--not-marker',
        action='store_true',
        help="[design] steer all decoded tokens except marker bracket spans ([...]); mutually exclusive with --marker-only",
    )

    args = parser.parse_args()
    if args.marker_only and args.not_marker:
        parser.error("--marker-only and --not-marker cannot be used together")
    is_api_model = args.model_name in API_MODELS
    if args.noref:
        assert args.layers is not None and len(args.layers) == 1, "--noref requires exactly one --layers value"
        assert args.multipliers is not None and len(args.multipliers) == 1, "--noref requires exactly one --multipliers value"
        assert abs(float(args.multipliers[0]) - 0.0) < 1e-6, "--noref requires --multipliers 0.0"
    if is_api_model:
        args.model_size = "api"
        if args.layers is None:
            args.layers = [0]
        if args.multipliers is None:
            args.multipliers = [0.0]

    steering_settings = SteeringSettings()
    steering_settings.type = args.type
    steering_settings.system_prompt = args.system_prompt
    steering_settings.override_vector = args.override_vector
    steering_settings.override_vector_model = args.override_vector_model
    steering_settings.use_base_model = args.use_base_model
    steering_settings.model_name = args.model_name
    steering_settings.model_size = args.model_size
    steering_settings.override_model_weights_path = args.override_model_weights_path

    all_inference_failures: List[Dict[str, Any]] = []
    for behavior in args.behaviors:
        steering_settings.behavior = behavior
        all_inference_failures.extend(
            test_steering(
                layers=args.layers,
                multipliers=args.multipliers,
                settings=steering_settings,
                overwrite=args.overwrite,
                resume=args.resume,
                model_name=args.model_name,
                model_size=args.model_size,
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
