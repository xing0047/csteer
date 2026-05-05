prompt_template = {
    "image_mc": {
        "pre_prompt": "Based on the image, select the best answer to the following multiple-choice question. In both the question and options, specific objects are represented using the format [ID] (e.g., '[1]'', '[2]''. Respond with only the letter (A, B, C, or D) of the correct option.",
        "post_prompt": "The best answer is: "
    },
    "gar_mc": {
        "pre_prompt": "Based on the image, select the best answer to the following multiple-choice question. In both the question and options, specific objects are represented using the format [ID] (e.g., '[1]'', '[2]''. Respond with only the letter (A, B, C, or D) of the correct option.",
        "post_prompt": "The best answer is: "
    },

    "image_think_mc": {
        "pre_prompt": "Based on the image, reason and answer the multiple-choice question. In both the question and options, specific objects are represented using the format [ID] (e.g., '[REF]'', '[A]''. Respond with only the letter (A, B, C, or D) of the correct option. provide your reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags.",
        "post_prompt": "Your answer: "
    },

    "blink_image_mc": {
        "pre_prompt": "Based on the image, select the best answer to the following multiple-choice question. In both the question and options, specific objects are represented using the format [ID] (e.g., '[REF]'', '[A]''. Respond with only the letter (A, B, C, or D) of the correct option.",
        "post_prompt": "The best answer is: "
    },

    "blink_image_think_mc": {
        "pre_prompt": "Based on the image, reason and answer the multiple-choice question. In both the question and options, specific objects are represented using the format [ID] (e.g., '[REF]'', '[A]''. Respond with only the letter (A, B, C, or D) of the correct option. provide your reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags.",
        "post_prompt": "Your answer: "
    },

    "cvbench_image_mc": {
        "pre_prompt": "Based on the image, select the best answer to the following multiple-choice question. In both the question and options, specific objects are represented using the format [ID] (e.g., '[REF]'', '[A]''. Respond with only the letter (A, B, C, or D) of the correct option.",
        "post_prompt": "The best answer is: "
    },

    "cvbench_image_think_mc": {
        "pre_prompt": "Based on the image, reason and answer the multiple-choice question. In both the question and options, specific objects are represented using the format [ID] (e.g., '[REF]'', '[A]''. Respond with only the letter (A, B, C, or D) of the correct option. provide your reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags.",
        "post_prompt": "Your answer: "
    },

    "image_oe": {
        "pre_prompt": "# Task Definition:\nYou are an expert in image analysis. In this task, you will receive an image, and your task is to answer the given question based on the image content.\n# Guidelines and Rules:\n- Object References: In the image, each object has a unique ID. Use this ID in your response to specify objects, formatted as [ID] for a single object (e.g., '[1]'', '[2]''). The IDs in the images and questions match directly.",
        "post_prompt": "Based on the input image, please answer the question: "
    },
    "gar_simple": {
        "pre_prompt": "# Task Definition:\nYou are an expert in image analysis. In this task, you will receive an image, and your task is to answer the given question based on the image content.\n# Guidelines and Rules:\n- Object References: In the image, each object has a unique ID. Use this ID in your response to specify objects, formatted as [ID] for a single object (e.g., '[1]'', '[2]''). The IDs in the images and questions match directly.",
        "post_prompt": "Based on the input image, please answer the question: "
    },
    "gar_detailed": {
        "pre_prompt": "# Task Definition:\nYou are an expert in image analysis. In this task, you will receive an image, and your task is to answer the given question based on the image content.\n# Guidelines and Rules:\n- Object References: In the image, each object has a unique ID. Use this ID in your response to specify objects, formatted as [ID] for a single object (e.g., '[1]'', '[2]''). The IDs in the images and questions match directly.",
        "post_prompt": "Based on the input image, please answer the question: "
    },

    # vLLM judge for compute_metric_gar_detailed (full user-message prefix; see JUDGE/evaluate_gar.py)
    "gar_detailed_eval": """
You are a language model expert. Your task is to evaluate the following model output based on the provided images, and subject, object, and relationship.

- subject_name: {subject_name}
- object_name: {object_name}
- predicate_name: {predicate_name}
- model_output: {model_output}

Task:
1. Check if the model output describes the {subject_name}. 
2. Check if the model output conveys the relationship between {subject_name} and {object_name} related to {predicate_name}.

Note:
- The first task only requires checking if {subject_name} is mentioned in the model output.
- The second task asks if the output conveys a relationship related to {predicate_name} between {subject_name} and {object_name}, even if different words or phrases are used.
- If both tasks are successfully completed, return "True" Otherwise, return "False"
- Do not output any reasoning. Do not perform correction. Please output only just one "True" or "False".
""",
    "gar_simple_eval": """
You are a language model expert. Your task is to evaluate the correctness of the model's output based on the provided ground truth and given masks.

- Ground truth: "{answer}"
- Model Output: "{model_output}"

Please determine if the model's output conveys the same meaning as the provided ground truth. If the output is semantically correct, return "True", otherwise return "False".

Attention:
1. The ground truth and model output do not need to match exactly, as long as they convey the same meaning. Synonyms and different phrasings are acceptable.

2. Do not output any reasoning. Do not perform correction. Please output only "True" or "False".
""",
    "vip_image_oe": {
        "pre_prompt": "# Task Definition:\nYou are an expert in image analysis. In this task, you will receive an image, and your task is to answer the given question based on the image content.\n# Guidelines and Rules:\n- Object References: In the image, each object has a unique ID. Use this ID in your response to specify objects, formatted as [ID] for a single object (e.g., '[red box]'', '[yellow box]''). The IDs in the images and questions match directly.",
        "post_prompt": "Based on the input image, please answer the question: "
    },

    "gar_image_simple_oe": {
        "pre_prompt": '''
# Task Definition:
You are an expert in image analysis. In this task, you will receive an image, and your task is to answer the given question based on the image content.
# Guidelines and Rules:
- Object References: In the image, each object is surrounded by a box and has an unique ID. Use this ID in your response to specify objects, formatted as [ID] for a single object (e.g., “[8]”) or as [ID1] [ID2] ... for multiple objects, such as “[3] [4] [5]”. Avoid commas, ranges, or phrases like “[1, 2, 3]” or “[1] to [3]”. The IDs in the images and questions match directly.
# Relation:
When describing spatial relations among objects, please consider multiple perspectives, including left-or-right, front-or-back, and other potential relations.
# Output Instructions:
Please first briefly recognize the referred objects or regions, then answer.
# Examples:
[0] is a person with a red hat who sits next to [1], a bird.
[1] is a cow standing on grass, in front of [0], a person taking photos for [1].''',
        "post_prompt": "Based on the input image, please answer the question: "
    },
    
    "video_mc": {
        "pre_prompt": 'Based on the video, select the best answer to the following multiple-choice question. In both the question and options, specific objects are represented using the format [ID] (e.g., "[1]", "[2]"), and time references are shown using the format <timestamp> (e.g., "at <6>" or "during <7>-<8>"). Respond with only the letter (A, B, C, or D) of the correct option.',
        "post_prompt": "The best answer is: "
    },

    "video_oe": {
        "pre_prompt": '# Task Definition:\nYou are an expert in video analysis. In this task, you will receive a series of frames as a video, and your task is to answer the given questions based on the video content.\n# Input Format:\nThere are serveral images inputs as video frames. Each frame can be referenced by its timestamp (indicating when it appears in the video). For example, the first frame can be referred to as <1>.\n# Guidelines and Rules:\n- Object References: Each object has a unique ID. Use this ID in your response to specify objects, formatted as [ID] for a single object (e.g., “[8]”) or as [ID1] [ID2] ... for multiple objects, such as “[3] [4] [5]”. Avoid commas, ranges, or phrases like “[1, 2, 3]” or “[1] to [3]”. The IDs in the images and questions match directly.\n- Time References: Use timestamps to indicate moments or intervals in the video. For a specific moment, format as <timestamp> (e.g., “at <3>”). For an interval, use <start_timestamp>-<end_timestamp> (e.g., “during <5>-<7>”). Always enclose timestamps in <>.',
        "post_prompt": "Based on the input video, please answer the question: "
    },

    "gpt_eval": {
        # image split
        "image": """
# Task Description:
You are an expert evaluator tasked with scoring the accuracy of responses to open-ended questions. You will be provided with a set of questions, each with a corresponding ground-truth answer, as well as responses from a tester. Your job is to assess the accuracy of each response and provide a score between 0 and 1.
# Guidelines:
- Score Range: Your score for each test item must be between 0 and 1. A higher score means more correctness. Choose from the following: 0 (completely incorrect), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 (completely correct)
- For each test item, consider the question, the ground-truth answer, and the tester’s response together to determine correctness.
- Objects in questions and answers may be referenced using the format [ID] (e.g., [1], [2]). Ensure that any objects referenced in the tester’s response match correctly with the ground-truth answer.
# Input Format:
The input is a set of test items to be scored, where each item includes: 
- `question`; 
- `ground truth answer for the question`; 
- `response from the tester`.
Now, let's begin the evaluation, here are the input test items: 
""",

        # video split
        "video": """
# Task Description:
You are an expert evaluator tasked with scoring the accuracy of responses to open-ended questions. You will be provided with a set of questions, each with a corresponding ground-truth answer, as well as responses from a tester. Your job is to assess the accuracy of each response and provide a score between 0 and 1.
# Guidelines:
- Score Range: Your score for each test item must be between 0 and 1. A higher score means more correctness. Choose from the following: 0 (completely incorrect), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 (completely correct)
- For each test item, consider the question, the ground-truth answer, and the tester’s response together to determine correctness.
- Objects in questions and answers may be referenced using the format [ID] (e.g., [1], [2]). Ensure that any objects referenced in the tester’s response match correctly with the ground-truth answer.
- Time points may be indicated with <timestamp> (e.g., <1>), and time intervals with <start_timestamp>-<end_timestamp> (e.g., <3>-<5>). Verify that the tester’s response includes accurate time expressions.
# Input Format:
The input is a set of test items to be scored, where each item includes: 
- `question`; 
- `ground truth answer for the question`; 
- `response from the tester`.
Now, let's begin the evaluation, here are the input test items:
"""
    },
    "mc_think_extract": {
        "system": """You extract the multiple-choice letter from a vision-language model's full output.
The output may contain chain-of-thought, optional reasoning wrapped in special tags, and a final answer.
Rules:
- Prefer the letter inside <answer>...</answer> if present (case-insensitive).
- Otherwise infer the model's chosen option (A, B, C, or D) from the conclusion only; ignore distractor letters in reasoning.
- Respond with exactly one uppercase letter: A, B, C, or D. If truly impossible, respond with X.""",
        "user_template": """Full model output:
---
{model_output}
---
Reply with exactly one character: A, B, C, D, or X.""",
    },
}
