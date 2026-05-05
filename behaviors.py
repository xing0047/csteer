import os
import re
from typing import Literal, Optional
from utils.helpers import make_tensor_save_suffix
import json
import torch as t
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REFER = "refer"

ALL_BEHAVIORS = [
    REFER
]

VECTORS_PATH = os.path.join(BASE_DIR, "../VECTOR")
NORMALIZED_VECTORS_PATH = os.path.join(BASE_DIR, "../NORMALIZED_VECTOR")
ANALYSIS_PATH = os.path.join(BASE_DIR, "analysis")
RESULTS_PATH = os.path.join(BASE_DIR, "../RESULT")
GENERATE_DATA_PATH = os.path.join(BASE_DIR, "datasets", "generate")
TEST_DATA_PATH = os.path.join(BASE_DIR, "../DATA")
RAW_DATA_PATH = os.path.join(BASE_DIR, "datasets", "raw")
ACTIVATIONS_PATH = os.path.join(BASE_DIR, "activations")
FINETUNE_PATH = os.path.join(BASE_DIR, "finetuned_models")
GAR_DATA_DIR = os.path.join(BASE_DIR, TEST_DATA_PATH , "GAR")
gar_image_mc_JSON = os.path.join(GAR_DATA_DIR, "GAR-Bench-VQA.json")
GAR_SIMPLE_JSON = os.path.join(GAR_DATA_DIR, "GAR-Bench-Caption-Simple.json")
GAR_DETAILED_JSON = os.path.join(GAR_DATA_DIR, "GAR-Bench-Caption-Detailed.json")
gar_image_mc_IMG_DIR = os.path.join(GAR_DATA_DIR, "vqa-images")
GAR_SIMPLE_IMG_DIR = os.path.join(GAR_DATA_DIR, "simple-images")
GAR_DETAILED_IMG_DIR = os.path.join(GAR_DATA_DIR, "detailed-images")

# Static dataset file paths (prefer constants over get_*_path() functions)
VIP_BENCH_QUESTIONS_JSONL = os.path.join(TEST_DATA_PATH, "ViP-Bench", "bbox", "questions.jsonl")
VIP_BENCH_META_JSON = os.path.join(TEST_DATA_PATH, "ViP-Bench", "vip-bench-meta-data.json")

INST_IT_IMAGE_MC_JSON = os.path.join(TEST_DATA_PATH, "Inst-It-Bench", "image_multi_choices.json")
INST_IT_IMAGE_OE_JSON = os.path.join(TEST_DATA_PATH, "Inst-It-Bench", "image_open_ended.json")
INST_IT_IMAGE_META_JSON = os.path.join(TEST_DATA_PATH, "Inst-It-Bench", "image_instance_captions_masks.json")

INST_IT_VIDEO_MC_JSON = os.path.join(TEST_DATA_PATH, "Inst-It-Bench", "video_multi_choices.json")
INST_IT_VIDEO_OE_JSON = os.path.join(TEST_DATA_PATH, "Inst-It-Bench", "video_open_ended.json")
INST_IT_VIDEO_META_JSON = os.path.join(TEST_DATA_PATH, "Inst-It-Bench", "video_instance_captions_masks.json")

# ---------------------------------------------------------------------------
# Raw (no-overlay) asset roots — placeholders. Set to an absolute path when you
# have the original images; while None, `noref=True` keeps the same paths as the
# annotated/default layout (current behavior).
# Keys match get_*_data() defaults: gar_image_mc / gar_simple default noref=True;
# gar_detail, blink, vip, inst_it default noref=False.
# ---------------------------------------------------------------------------
# ViP-Bench: raw images (see RAW.md); filenames from metadata["image_source"] when noref=True.
NOREF_VIP_IMAGE_DIR = os.path.join(TEST_DATA_PATH, "ViP-Bench", "source_image")
NOREF_PLACEHOLDER_GAR_VQA_IMAGE_DIR: Optional[str] = None
NOREF_PLACEHOLDER_GAR_SIMPLE_IMAGE_DIR: Optional[str] = None
# GAR-detailed OE (RAW.md): no-refer originals live in ../DATA/GAR/images
NOREF_GAR_DETAILED_IMAGE_DIR = os.path.join(TEST_DATA_PATH, "GAR", "images")
# BLINK: MsDataset rows — set when you have a local mirror or alternate loader.
NOREF_PLACEHOLDER_BLINK_USE_RAW: bool = False
# Inst-It image: HF dataset — set when you point image column to raw files.
NOREF_PLACEHOLDER_INST_IT_IMAGE_USE_RAW: bool = False
# Inst-It video: frame folders under ../DATA/Inst-It-Bench/<video_path>
NOREF_PLACEHOLDER_INST_IT_VIDEO_PATH_PREFIX: Optional[str] = None

# Inst-It (RAW.md): no-refer raw images live in images_raw/, refer images in images_vpt/
INST_IT_IMAGE_RAW_DIR = os.path.join(TEST_DATA_PATH, "Inst-It-Bench", "images_raw")
INST_IT_IMAGE_REFER_DIR = os.path.join(TEST_DATA_PATH, "Inst-It-Bench", "images_vpt")

# Inst-It video (RAW.md): frame folders under videos_raw/ (no refer) and videos_vpt/ (refer)
INST_IT_VIDEO_RAW_PREFIX = "videos_raw"
INST_IT_VIDEO_REFER_PREFIX = "videos_vpt"


def _raw_path_if_set(raw_root: Optional[str], basename: str, default_path: str, noref: bool) -> str:
    if noref and raw_root:
        return os.path.join(raw_root, basename)
    return default_path


def get_vector_dir(behavior: str, model_name: str, model_size: str, output_dir: str, normalized=False) -> str:
    return os.path.join(NORMALIZED_VECTORS_PATH if normalized else VECTORS_PATH, behavior, f"{model_name}_{model_size}", output_dir)

def get_vector_path(behavior: str, model_name: str, model_size: str, output_dir: str, layer, normalized=False) -> str:
    return os.path.join(
        get_vector_dir(behavior, model_name, model_size, output_dir, normalized=normalized),
        f"vec_layer_{make_tensor_save_suffix(layer, model_name, model_size)}.pt",
    )

def get_results_dir(behavior: str, model_name: str, model_size: str, output_dir: str) -> str:
    return os.path.join(RESULTS_PATH, behavior, f"{model_name}_{model_size}", output_dir)

def get_raw_data_path(behavior: str) -> str:
    pdb.set_trace()  # todo
    return os.path.join(RAW_DATA_PATH, behavior, "dataset.json")

def get_open_ended_data_path(behavior: str) -> str:
    pdb.set_trace()  # todo
    return os.path.join(TEST_DATA_PATH, behavior, "test_dataset_open_ended.json")

def get_gar_simple_images_dir() -> str:
    return os.path.join(TEST_DATA_PATH, "GAR", "simple-images")

def get_gar_detailed_images_dir() -> str:
    return os.path.join(TEST_DATA_PATH, "GAR", "detailed-images")

def get_analysis_dir(behavior: str) -> str:
    pdb.set_trace()  # todo
    return os.path.join(ANALYSIS_PATH, behavior)

def get_activations_dir(behavior: str) -> str:
    return os.path.join(ACTIVATIONS_PATH, behavior)

def get_activations_path(
    behavior: str, layer, model_name: str, model_size: str, pos_or_neg: Literal["pos", "neg"]
) -> str:
    return os.path.join(
        get_activations_dir(behavior),
        f"activations_{pos_or_neg}_{make_tensor_save_suffix(layer, model_name, model_size)}.pt",
    )

def get_open_ended_test_data(behavior):
    pdb.set_trace()  # todo
    with open(get_open_ended_data_path(behavior), "r") as f:
        data = json.load(f)
    return data

def get_vip_bench_data(noref: bool = False):
    with open(VIP_BENCH_QUESTIONS_JSONL, "r") as f:
        data = [json.loads(line.strip()) for line in f.readlines()]
        metadata = json.load(open(VIP_BENCH_META_JSON))
        out = []
        for item, metaditem_key in zip(data, metadata):
            assert item["image"] == metadata[metaditem_key]["image"]
            item["text"] = re.sub(r"(\w+) rectangle", r"[\1 rectangle]", item["text"])
            bbox_path = os.path.join("..", "DATA", "ViP-Bench", "bbox", "images", item["image"])
            if noref:
                # RAW.md: originals under NOREF_VIP_IMAGE_DIR; filename from metadata["image_source"].
                meta_entry = metadata[metaditem_key]
                source_name = os.path.basename(meta_entry.get("image_source", item["image"]))
                image_path = os.path.join(NOREF_VIP_IMAGE_DIR, source_name)
            else:
                image_path = bbox_path
            out.append({
                "question_id": metaditem_key,
                "image_id": item["image"].replace(".png", ""),
                "image_path": image_path,
                "question": item["text"],
                "answer": metadata[metaditem_key]["answer"],
                "capability": metadata[metaditem_key]["capability"]
            })
    return out

def get_dlc_bench_data():
    pdb.set_trace()  # todo
    with open(VIP_BENCH_QUESTIONS_JSONL, "r") as f:
        data = [json.loads(line.strip()) for line in f.readlines()]
    return data

def get_blink_image_mc_data(noref: bool = False, subsets: Optional[list] = None):
    from modelscope.msdatasets import MsDataset
    from tqdm import tqdm
    BLINK_SUBSETS = [
        "Visual_Correspondence",
        "Relative_Reflectance",
        "Relative_Depth",
        "Object_Localization",
        "Semantic_Correspondence",
        "Functional_Correspondence"
    ]
    if subsets is not None:
        BLINK_SUBSETS = subsets
    out = []
    for subset in tqdm(BLINK_SUBSETS, position=0, leave=False):
        ds = MsDataset.load("evalscope/BLINK", subset_name=subset, split="val")
        for ex in tqdm(ds, position=1, leave=False):
            # noref default False; raw BLINK paths: set NOREF_PLACEHOLDER_BLINK_USE_RAW and load from your mirror here.
            images = [image for image in [ex["image_1"], ex["image_2"], ex["image_3"], ex["image_4"]] if image is not None]
            if subset == "Visual_Correspondence":
                assert "the reference point" in ex["question"]
                assert len(ex["choices"]) == 4
                ex["question"] = ex["question"].replace("the reference point", "the reference point [REF]")
                ex["choices"] = [re.sub(r'Point (\w+)', r'Point [\1]', choice) for choice in ex["choices"]]
                out.append(
                    {
                        "question_id": ex["idx"],
                        "image_id": ex["idx"],
                        "type": ex["sub_task"],
                        "pil_images": images,
                        "question": ex["question"],
                        "choices": ex["choices"],
                        "answer": ex.get("answer", None).strip('()')
                    }
                )
            elif subset == "Relative_Reflectance":
                assert len(ex["choices"]) == 3
                assert "A is darker" in ex["choices"][0]
                assert "B is darker" in ex["choices"][1]
                ex["choices"][0] = ex["choices"][0].replace("A", "Point [A]")
                ex["choices"][1] = ex["choices"][1].replace("B", "Point [B]")
                out.append(
                    {
                        "question_id": ex["idx"],
                        "image_id": ex["idx"],
                        "type": ex["sub_task"],
                        "pil_images": images,
                        "question": ex["question"],
                        "choices": ex["choices"],
                        "answer": ex.get("answer", None).strip('()')
                    }
                )
            elif subset == "Relative_Depth":
                assert len(ex["choices"]) == 2
                assert "A is closer" in ex["choices"][0]
                assert "B is closer" in ex["choices"][1]
                ex["choices"][0] = ex["choices"][0].replace("A", "Point [A]")
                ex["choices"][1] = ex["choices"][1].replace("B", "Point [B]")
                out.append(
                    {
                        "question_id": ex["idx"],
                        "image_id": ex["idx"],
                        "type": ex["sub_task"],
                        "pil_images": images,
                        "question": ex["question"],
                        "choices": ex["choices"],
                        "answer": ex.get("answer", None).strip('()')
                    }
                )
            elif subset == "Object_Localization":
                assert len(ex["choices"]) == 2
                assert "Box A" in ex["choices"][0]
                assert "Box B" in ex["choices"][1]
                ex["choices"][0] = ex["choices"][0].replace("Box A", "Box [A]")
                ex["choices"][1] = ex["choices"][1].replace("Box B", "Box [B]")
                out.append(
                    {
                        "question_id": ex["idx"],
                        "image_id": ex["idx"],
                        "type": ex["sub_task"],
                        "pil_images": images,
                        "question": ex["question"],
                        "choices": ex["choices"],
                        "answer": ex.get("answer", None).strip('()')
                    }
                )
            elif subset == "Semantic_Correspondence":
                assert "the reference point" in ex["question"]
                assert len(ex["choices"]) == 4
                ex["question"] = ex["question"].replace("the reference point", "the reference point [REF]")
                ex["choices"] = [re.sub(r'Point (\w+)', r'Point [\1]', choice) for choice in ex["choices"]]
                out.append(
                    {
                        "question_id": ex["idx"],
                        "image_id": ex["idx"],
                        "type": ex["sub_task"],
                        "pil_images": images,
                        "question": ex["question"],
                        "choices": ex["choices"],
                        "answer": ex.get("answer", None).strip('()')
                    }
                )
            elif subset == "Functional_Correspondence":
                assert "the reference point" in ex["question"]
                assert len(ex["choices"]) == 4
                ex["question"] = ex["question"].replace("the reference point", "the reference point [REF]")
                ex["choices"] = [re.sub(r'Point (\w+)', r'Point [\1]', choice) for choice in ex["choices"]]
                out.append(
                    {
                        "question_id": ex["idx"],
                        "image_id": ex["idx"],
                        "type": ex["sub_task"],
                        "pil_images": images,
                        "question": ex["question"],
                        "choices": ex["choices"],
                        "answer": ex.get("answer", None).strip('()')
                    }
                )
    return out

def get_cvbench_image_mc_data(noref: bool = False):
    from modelscope.msdatasets import MsDataset
    from tqdm import tqdm
    CVBENCH_SUBSETS = [
        "Relation",
        "Depth",
        "Distance"
    ]
    ds = MsDataset.load('comefly/cvbench', split='test')
    out = []
    for ex in tqdm(ds, leave=False):
        if ex["task"] not in CVBENCH_SUBSETS:
            continue
        if ex["task"] ==  "Relation":
            if "annotated" not in ex["prompt"]:
                continue
            ex["prompt"] = re.sub(r"annotated by the (\w+) box", r"[\1 box]", ex["prompt"])
        elif ex["task"] == "Depth":
            ex["prompt"] = re.sub(r"highlighted by a (\w+) box", r"[\1 box]", ex["prompt"])
        elif ex["task"] == "Distance":
            ex["prompt"] = re.sub(r"highlighted by a (\w+) box", r"[\1 box]", ex["prompt"])

        out.append({
            "question_id": ex["idx"],
            "image_id": ex["idx"],
            "source_filename": ex["source_filename"],
            "type": ex["task"],
            "pil_image": ex["image"],
            "question": ex["prompt"],
            "answer": ex.get("answer", None).strip('()')
        })
    
        assert "[red box]" in out[-1]["question"]
    return out

def get_inst_it_image_mc_data(noref: bool = False):
    # RAW.md: items contain image_path like "images_vpt/001.jpg".
    # - noref=False (default): load images_vpt/<basename>
    # - noref=True:            load images_raw/<basename>
    with open(INST_IT_IMAGE_MC_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = []
    for ex in data:
        rel_path = ex.get("image_path", "")
        basename = os.path.basename(rel_path) if isinstance(rel_path, str) else ""
        img_dir = INST_IT_IMAGE_RAW_DIR if noref else INST_IT_IMAGE_REFER_DIR
        img_path = os.path.join(img_dir, basename)
        pil_image = Image.open(img_path).convert("RGB")

        opts = ex.get("options", {}) or {}
        out.append(
            {
                "question_id": str(ex.get("question_id", "")),
                "image_id": str(ex.get("image_id", "")),
                "image_path": img_path,
                "pil_image": pil_image,
                "question": ex.get("question", ""),
                "choice_a": opts.get("A", ""),
                "choice_b": opts.get("B", ""),
                "choice_c": opts.get("C", ""),
                "choice_d": opts.get("D", ""),
                "answer": ex.get("answer", None),
                "meta_info": ex.get("meta_info", {}),
            }
        )
    return out

def get_inst_it_image_oe_data(noref: bool = False):
    # RAW.md: items contain image_path like "images_vpt/001.jpg".
    # - noref=False (default): load images_vpt/<basename>
    # - noref=True:            load images_raw/<basename>
    with open(INST_IT_IMAGE_OE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = []
    for ex in data:
        rel_path = ex.get("image_path", "")
        basename = os.path.basename(rel_path) if isinstance(rel_path, str) else ""
        img_dir = INST_IT_IMAGE_RAW_DIR if noref else INST_IT_IMAGE_REFER_DIR
        img_path = os.path.join(img_dir, basename)
        pil_image = Image.open(img_path).convert("RGB")

        out.append(
            {
                "question_id": str(ex.get("question_id", "")),
                "image_id": str(ex.get("image_id", "")),
                "image_path": img_path,
                "pil_image": pil_image,
                "question": ex.get("question", ""),
                "answer": ex.get("answer", None),
                "meta_info": ex.get("meta_info", {}),
            }
        )
    return out

def get_inst_it_video_mc_data(noref: bool = False):
    # RAW.md: use video_path under videos_vpt/ (refer) or videos_raw/ (no refer).
    with open(INST_IT_VIDEO_MC_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(INST_IT_VIDEO_META_JSON, "r", encoding="utf-8") as meta_f:
        metadata = json.load(meta_f)

    prefix = INST_IT_VIDEO_RAW_PREFIX if noref else INST_IT_VIDEO_REFER_PREFIX
    for item in data:
        video_id = str(item.get("video_id", ""))
        segs = metadata.get("segmentations", {}).get(video_id, {})
        item["annotations"] = {frame: segs[frame] for frame in segs}

        raw_vp = item.get("video_path", "")
        basename_dir = os.path.basename(os.path.normpath(raw_vp)) if isinstance(raw_vp, str) else ""
        item["video_path"] = f"{prefix}/{basename_dir}"

    return data

def get_inst_it_video_oe_data(noref: bool = False):
    # RAW.md: use video_path under videos_vpt/ (refer) or videos_raw/ (no refer).
    with open(INST_IT_VIDEO_OE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(INST_IT_VIDEO_META_JSON, "r", encoding="utf-8") as meta_f:
        metadata = json.load(meta_f)

    prefix = INST_IT_VIDEO_RAW_PREFIX if noref else INST_IT_VIDEO_REFER_PREFIX
    for item in data:
        video_id = str(item.get("video_id", ""))
        segs = metadata.get("segmentations", {}).get(video_id, {})
        item["annotations"] = {frame: segs[frame] for frame in segs}

        raw_vp = item.get("video_path", "")
        basename_dir = os.path.basename(os.path.normpath(raw_vp)) if isinstance(raw_vp, str) else ""
        item["video_path"] = f"{prefix}/{basename_dir}"

    return data

def get_gar_image_mc_data(max_num: Optional[int] = None, noref: bool = True) -> list:
    with open(gar_image_mc_JSON, "r") as f:
        raw_data = json.load(f)

    items = []
    for idx, ann in enumerate(raw_data):
        img_name = os.path.basename(ann["image"])
        default_image_path = os.path.join(gar_image_mc_IMG_DIR, img_name)
        image_path = _raw_path_if_set(
            NOREF_PLACEHOLDER_GAR_VQA_IMAGE_DIR, img_name, default_image_path, noref
        )
        pil_image = Image.open(image_path)

        items.append(
            {
                "sample_id": idx,
                "image_path": image_path,
                "pil_image": pil_image,
                "question": ann["question"],
                "choices": ann["choices"],
                "answer": ann.get("answer", ""),
                'type': ann['type'],
                "mask_rles": ann.get("mask_rles", []),
            }
        )
        if max_num is not None and len(items) >= max_num:
            break

    return items

def get_gar_caption_simple_data(max_num: Optional[int] = None, noref: bool = True) -> list:
    with open(GAR_SIMPLE_JSON, "r") as f:
        raw_data = json.load(f)

    items = []
    for idx, ann in enumerate(raw_data):
        img_name = os.path.basename(ann["image"])
        default_image_path = os.path.join(GAR_SIMPLE_IMG_DIR, img_name)
        image_path = _raw_path_if_set(
            NOREF_PLACEHOLDER_GAR_SIMPLE_IMAGE_DIR, img_name, default_image_path, noref
        )

        items.append(
            {
                "sample_id": idx,
                "image_path": image_path,
                "question": ann["question"],                
                "answer": ann.get("answer", ""),
                "mask_rles": ann.get("mask_rles", []),
            }
        )
        if max_num is not None and len(items) >= max_num:
            break

    return items

def get_gar_caption_detailed_data(max_num: Optional[int] = None, noref: bool = False) -> list:
    with open(GAR_DETAILED_JSON, "r") as f:
        raw_data = json.load(f)

    items = []
    question = "Describe [0] in detail, including the relationship with [1]."
    for idx, ann in enumerate(raw_data):
        img_name = os.path.basename(ann["image"])
        # RAW.md:
        # - refer (default): ../DATA/GAR/detailed-images/<img_name>
        # - no-refer:        ../DATA/GAR/images/<img_name>
        default_image_path = os.path.join(GAR_DETAILED_IMG_DIR, img_name)
        image_path = os.path.join(NOREF_GAR_DETAILED_IMAGE_DIR, img_name) if noref else default_image_path

        items.append(
            {
                "sample_id": idx,
                "image_path": image_path,
                "question": question,
                "subject_name": ann.get("subject_name", ""),
                "object_name": ann.get("object_name", ""),
                "predicate_name": ann.get("predicate_name", ""),
                "mask_rles": ann.get("mask_rles", []),
            }
        )
        if max_num is not None and len(items) >= max_num:
            break

    return items

def get_steering_vector(behavior, model_name, model_size, output_dir, layer, normalized=False):
    return t.load(get_vector_path(behavior, model_name, model_size, output_dir, layer, normalized=normalized))

def get_finetuned_model_path(
    behavior: str, pos_or_neg: Optional[Literal["pos", "neg"]], layer=None
) -> str:
    pdb.set_trace()  # todo
    if layer is None:
        layer = "all"
    return os.path.join(
        FINETUNE_PATH,
        f"{behavior}_{pos_or_neg}_finetune_{layer}.pt",
    )

def get_finetuned_model_results_path(
    behavior: str, pos_or_neg: Optional[Literal["pos", "neg"]], eval_type: str, layer=None
) -> str:
    pdb.set_trace()  # todo
    if layer is None:
        layer = "all"
    return os.path.join(
        RESULTS_PATH,
        f"{behavior}_{pos_or_neg}_finetune_{layer}_{eval_type}_results.json",
    )
