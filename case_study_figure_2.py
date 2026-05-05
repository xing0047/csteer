import json
from tqdm import tqdm

baseline_json="../RESULT/refer/qwen3vl_8b/inst_it_image_oe_qa/inst_it_image/04_rew_n1024/02_marker_only/02_judged/results_layer_00_multiplier_0.0_behavior_refer_type_inst_it_image_oe_qa_model_name_qwen3vl_model_size_8b_evaluated.json"
baseline_result=json.load(open(baseline_json))
steer_result_json_format="../RESULT/refer/qwen3vl_8b/inst_it_image_oe_qa/inst_it_image/04_rew_n1024/02_marker_only/02_judged/results_layer_{}_multiplier_1.0_behavior_refer_type_inst_it_image_oe_qa_model_name_qwen3vl_model_size_8b_evaluated.json"

matched = {}
for layer in tqdm(range(0, 36), position=0, leave=False):
    steer_result = json.load(open(steer_result_json_format.format(str(layer).zfill(2))))
    # compare score for every data pair and keep score difference
    for base_item_result, steer_item_result in tqdm(zip(baseline_result, steer_result), position=1, leave=False):
        assert base_item_result["question_id"] == steer_item_result["question_id"]
        question_id = base_item_result["question_id"]
        if question_id not in matched:
            matched[question_id] = [0. for _ in range(36)]
        matched[question_id][layer] = steer_item_result["score"] - base_item_result["score"]
sorted_matched = dict(sorted(matched.items(), key=lambda item: sum(item[1]), reverse=True))
json.dump(sorted_matched, open("../CASE/fig2_sorted_matched.json", "w"), indent=4)
