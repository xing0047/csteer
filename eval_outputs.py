"""
Unified evaluation entry for output JSON files.

Dataset-specific logic lives in eval_utils/*.
"""

from __future__ import annotations

import argparse
import json

from behaviors import VIP_BENCH_META_JSON, VIP_BENCH_QUESTIONS_JSONL
from eval_utils.eval_blink import eval_blink_subset, eval_cvbench
from eval_utils.eval_vip import eval_vip_bbox_lmdeploy
from eval_utils.evaluate_gar import (
    compute_metric_gar_detailed,
    compute_metric_gar_simple,
    compute_metric_mc_gar_vqa,
)
from eval_utils.evaluate_inst_it import (
    VLLMJudge,
    evaluate_mc_file,
    load_inst_it_image_mc_ground_truth,
    load_inst_it_image_oe_ground_truth,
    load_inst_it_video_mc_ground_truth,
    load_inst_it_video_oe_ground_truth,
    process_oe_json_file,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate output JSON files.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_vip = sub.add_parser("vip", help="ViP-Bench bbox open-ended (LLM judge, 0–1 scores).")
    p_vip.add_argument("--input", required=True, help="results_*.json from prompting_with_steering.py")
    p_vip.add_argument("--output", default=None, help="optional JSON metrics + grades")
    p_vip.add_argument("--base_url", default="http://127.0.0.1:23333/v1", help="OpenAI-compatible base URL")
    p_vip.add_argument("--model_name", default="Qwen/Qwen2.5-72B-Instruct-AWQ", help="Served model name (lmdeploy)")
    p_vip.add_argument("--meta_json", default=VIP_BENCH_META_JSON)
    p_vip.add_argument("--questions_jsonl", default=VIP_BENCH_QUESTIONS_JSONL)
    p_vip.add_argument("--api_key", default="EMPTY")
    p_vip.add_argument("--num_run", type=int, default=1)

    p_blink = sub.add_parser("blink", help="BLINK MC: Relative_Reflectance / Relative_Depth / Functional_Correspondence.")
    p_blink.add_argument("--input", required=True)
    p_blink.add_argument("--output", default=None, help="optional metrics JSON")
    p_blink.add_argument("--mode", default="blink_image_mc_qa", choices=["blink_image_mc_qa"])
    p_blink.add_argument("--subsets", nargs="*", default=None, help="override subset names")

    p_cv = sub.add_parser("cvbench", help="CV-Bench MC (full file).")
    p_cv.add_argument("--input", required=True)
    p_cv.add_argument("--output", default=None)
    p_cv.add_argument("--mode", default="cvbench_image_mc_qa", choices=["cvbench_image_mc_qa"])

    p_inst = sub.add_parser("inst_it", help="Inst-It-Bench: MC (no LLM) or OE (lmdeploy judge).")
    p_inst.add_argument("--mode", required=True, choices=["image_mc", "video_mc", "image_oe", "video_oe"])
    p_inst.add_argument("--input", required=True, help="one results JSON file")
    p_inst.add_argument("--output", default=None, help="MC scored JSON / OE judged JSON")
    p_inst.add_argument("--base_url", default="http://127.0.0.1:23333/v1")
    p_inst.add_argument("--model_name", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    p_inst.add_argument("--num_workers", type=int, default=8)
    p_inst.add_argument("--overwrite", action="store_true")

    p_gar = sub.add_parser("gar", help="GAR: MC or simple/detail judged metrics.")
    p_gar.add_argument("--kind", required=True, choices=["mc", "simple", "detailed"])
    p_gar.add_argument("--input", required=True)
    p_gar.add_argument("--output", default=None, help="metrics JSON path")
    p_gar.add_argument("--base_url", default="http://127.0.0.1:23333/v1")
    p_gar.add_argument("--model_name", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    p_gar.add_argument("--num_process", type=int, default=8)

    args = p.parse_args()

    if args.cmd == "vip":
        eval_vip_bbox_lmdeploy(
            args.input,
            args.output,
            args.base_url,
            args.model_name,
            args.meta_json,
            args.questions_jsonl,
            api_key=args.api_key,
            num_run=args.num_run,
        )
    elif args.cmd == "blink":
        eval_blink_subset(args.input, args.subsets, args.output, mode=args.mode)
    elif args.cmd == "cvbench":
        eval_cvbench(args.input, args.output, mode=args.mode)
    elif args.cmd == "inst_it":
        if args.mode.endswith("_mc"):
            if args.mode == "image_mc":
                gt_list = load_inst_it_image_mc_ground_truth()
                split = "image"
            else:
                gt_list = load_inst_it_video_mc_ground_truth()
                split = "video"
            out = args.output or args.input.replace(".json", "_mc_eval.json")
            evaluate_mc_file(args.input, out, gt_list, split=split, match_by_question_id=True)
            print(f"[inst_it {args.mode}] wrote {out}")
        else:
            gt_loaders = {
                "image_oe": load_inst_it_image_oe_ground_truth,
                "video_oe": load_inst_it_video_oe_ground_truth,
            }
            gt_list = gt_loaders[args.mode]()
            split = "image" if args.mode == "image_oe" else "video"
            out = args.output or args.input.replace(".json", "_evaluated.json")
            judge = VLLMJudge(base_url=args.base_url, model_name=args.model_name)
            process_oe_json_file(
                args.input,
                out,
                judge,
                num_workers=args.num_workers,
                overwrite=args.overwrite,
                gt_list=gt_list,
                split=split,
                match_by_question_id=True,
            )
            print(f"[inst_it {args.mode}] wrote {out}")
    elif args.cmd == "gar":
        with open(args.input, "r", encoding="utf-8") as f:
            preds = json.load(f)
        out = args.output or args.input.replace(".json", f"_{args.kind}_metrics.json")
        if args.kind == "mc":
            compute_metric_mc_gar_vqa(preds, out)
        elif args.kind == "simple":
            compute_metric_gar_simple(
                preds,
                out,
                base_url=args.base_url,
                model_name=args.model_name,
                num_process=args.num_process,
            )
        else:
            compute_metric_gar_detailed(
                preds,
                out,
                base_url=args.base_url,
                model_name=args.model_name,
                num_process=args.num_process,
            )
        print(f"[gar {args.kind}] wrote {out}")


if __name__ == "__main__":
    main()
"""
Unified evaluation entry for output JSON files.

Dataset-specific logic lives in eval_utils/*.
"""

from __future__ import annotations

import argparse
import json

from behaviors import VIP_BENCH_META_JSON, VIP_BENCH_QUESTIONS_JSONL
from eval_utils.eval_blink import eval_blink_subset, eval_cvbench
from eval_utils.eval_vip import eval_vip_bbox_lmdeploy
from eval_utils.evaluate_gar import (
    compute_metric_gar_detailed,
    compute_metric_gar_simple,
    compute_metric_mc_gar_vqa,
)
from eval_utils.evaluate_inst_it import (
    VLLMJudge,
    evaluate_mc_file,
    load_inst_it_image_mc_ground_truth,
    load_inst_it_image_oe_ground_truth,
    load_inst_it_video_mc_ground_truth,
    load_inst_it_video_oe_ground_truth,
    process_oe_json_file,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate output JSON files.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_vip = sub.add_parser("vip", help="ViP-Bench bbox open-ended (LLM judge, 0–1 scores).")
    p_vip.add_argument("--input", required=True, help="results_*.json from prompting_with_steering.py")
    p_vip.add_argument("--output", default=None, help="optional JSON metrics + grades")
    p_vip.add_argument("--base_url", default="http://127.0.0.1:23333/v1", help="OpenAI-compatible base URL")
    p_vip.add_argument("--model_name", default="Qwen/Qwen2.5-72B-Instruct-AWQ", help="Served model name (lmdeploy)")
    p_vip.add_argument("--meta_json", default=VIP_BENCH_META_JSON)
    p_vip.add_argument("--questions_jsonl", default=VIP_BENCH_QUESTIONS_JSONL)
    p_vip.add_argument("--api_key", default="EMPTY")
    p_vip.add_argument("--num_run", type=int, default=1)

    p_blink = sub.add_parser("blink", help="BLINK MC: Relative_Reflectance / Relative_Depth / Functional_Correspondence.")
    p_blink.add_argument("--input", required=True)
    p_blink.add_argument("--output", default=None, help="optional metrics JSON")
    p_blink.add_argument("--mode", default="blink_image_mc_qa", choices=["blink_image_mc_qa"])
    p_blink.add_argument("--subsets", nargs="*", default=None, help="override subset names")

    p_cv = sub.add_parser("cvbench", help="CV-Bench MC (full file).")
    p_cv.add_argument("--input", required=True)
    p_cv.add_argument("--output", default=None)
    p_cv.add_argument("--mode", default="cvbench_image_mc_qa", choices=["cvbench_image_mc_qa"])

    p_inst = sub.add_parser("inst_it", help="Inst-It-Bench: MC (no LLM) or OE (lmdeploy judge).")
    p_inst.add_argument("--mode", required=True, choices=["image_mc", "video_mc", "image_oe", "video_oe"])
    p_inst.add_argument("--input", required=True, help="one results JSON file")
    p_inst.add_argument("--output", default=None, help="MC scored JSON / OE judged JSON")
    p_inst.add_argument("--base_url", default="http://127.0.0.1:23333/v1")
    p_inst.add_argument("--model_name", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    p_inst.add_argument("--num_workers", type=int, default=8)
    p_inst.add_argument("--overwrite", action="store_true")

    p_gar = sub.add_parser("gar", help="GAR: MC or simple/detail judged metrics.")
    p_gar.add_argument("--kind", required=True, choices=["mc", "simple", "detailed"])
    p_gar.add_argument("--input", required=True)
    p_gar.add_argument("--output", default=None, help="metrics JSON path")
    p_gar.add_argument("--base_url", default="http://127.0.0.1:23333/v1")
    p_gar.add_argument("--model_name", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    p_gar.add_argument("--num_process", type=int, default=8)

    args = p.parse_args()

    if args.cmd == "vip":
        eval_vip_bbox_lmdeploy(
            args.input,
            args.output,
            args.base_url,
            args.model_name,
            args.meta_json,
            args.questions_jsonl,
            api_key=args.api_key,
            num_run=args.num_run,
        )
    elif args.cmd == "blink":
        eval_blink_subset(args.input, args.subsets, args.output, mode=args.mode)
    elif args.cmd == "cvbench":
        eval_cvbench(args.input, args.output, mode=args.mode)
    elif args.cmd == "inst_it":
        if args.mode.endswith("_mc"):
            if args.mode == "image_mc":
                gt_list = load_inst_it_image_mc_ground_truth()
                split = "image"
            else:
                gt_list = load_inst_it_video_mc_ground_truth()
                split = "video"
            out = args.output or args.input.replace(".json", "_mc_eval.json")
            evaluate_mc_file(args.input, out, gt_list, split=split, match_by_question_id=True)
            print(f"[inst_it {args.mode}] wrote {out}")
        else:
            gt_loaders = {
                "image_oe": load_inst_it_image_oe_ground_truth,
                "video_oe": load_inst_it_video_oe_ground_truth,
            }
            gt_list = gt_loaders[args.mode]()
            split = "image" if args.mode == "image_oe" else "video"
            out = args.output or args.input.replace(".json", "_evaluated.json")
            judge = VLLMJudge(base_url=args.base_url, model_name=args.model_name)
            process_oe_json_file(
                args.input,
                out,
                judge,
                num_workers=args.num_workers,
                overwrite=args.overwrite,
                gt_list=gt_list,
                split=split,
                match_by_question_id=True,
            )
            print(f"[inst_it {args.mode}] wrote {out}")
    elif args.cmd == "gar":
        with open(args.input, "r", encoding="utf-8") as f:
            preds = json.load(f)
        out = args.output or args.input.replace(".json", f"_{args.kind}_metrics.json")
        if args.kind == "mc":
            compute_metric_mc_gar_vqa(preds, out)
        elif args.kind == "simple":
            compute_metric_gar_simple(
                preds,
                out,
                base_url=args.base_url,
                model_name=args.model_name,
                num_process=args.num_process,
            )
        else:
            compute_metric_gar_detailed(
                preds,
                out,
                base_url=args.base_url,
                model_name=args.model_name,
                num_process=args.num_process,
            )
        print(f"[gar {args.kind}] wrote {out}")


if __name__ == "__main__":
    main()
