import json
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, choices=["internvl2_5", "internvl3"], required=True)
parser.add_argument("--model_size", type=str, choices=["2b", "8b", "26b", "38b"], required=True)
parser.add_argument("--layers", nargs="+", type=int, required=True)
args = parser.parse_args()

###### define the type of your visual prompt  ######
vipbench_split = 'bbox' # 'bbox'
vipbench_path = "datasets/test/vipbench" #  "/path/to/vip-bench"
###### change your model name ######
model = f"{args.model_name}_{args.model_size}"
result_path = os.path.join("results", "refer")
grade_path = os.path.join("grades", "refer")
num_run = 1 # we set it as 5 in the paper

results_files = [f"results_layer=00_multiplier=0.0_behavior=refer_type=vip_image_oe_qa_model_name={args.model_name}_model_size={args.model_size}.json"]
for layer in args.layers:
    for multiplier in [1.0]:
        results_files.append(f"results_layer={str(layer).zfill(2)}_multiplier={multiplier}_behavior=refer_type=vip_image_oe_qa_model_name={args.model_name}_model_size={args.model_size}.json")

from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-123456",
)
gpt_model = "Qwen/Qwen2.5-72B-Instruct"


prompt = """Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. <AND> in the ground truth means it is totally right only when all elements in the ground truth are present in the prediction, and <OR> means it is totally right when any one element in the ground truth is present in the prediction. The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Just complete the last space of the correctness score. Output the correctness score from 0.0 to 1.0 ONLY.

Question | Ground truth | Prediction | Correctness
--- | --- | --- | ---
What is x in the equation within the yellow rectangle? | -1 <AND> -5 | x = 3 | 0.0
What is x in the equation within the yellow rectangle? | -1 <AND> -5 | x = -1 | 0.5
What is x in the equation within the yellow rectangle? | -1 <AND> -5 | x = -5 | 0.5
What is x in the equation within the red rectangle? | -1 <AND> -5 | x = -5 or 5 | 0.5
What is x in the equation within the orange rectangle? | -1 <AND> -5 | x = -1 or x = -5 | 1.0
What is x in the equation within the orange rectangle? | -1 <AND> -5 | x = -1 or x = -5 | 1.0
Can you explain this meme within the blue rectangle? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme talks about Iceland and Greenland. It's pointing out that despite their names, Iceland is not very icy and Greenland isn't very green. | 0.4
Can you explain this meme within the blue rectangle? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme is using humor to point out the misleading nature of Iceland's and Greenland's names. Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow. The text 'This is why I have trust issues' is a playful way to suggest that these contradictions can lead to distrust or confusion. The humor in this meme is derived from the unexpected contrast between the names of the countries and their actual physical characteristics. | 1.0
"""

# load metadata
# Download ViP-Bench from https://huggingface.co/datasets/mucai/ViP-Bench
use_sub_set = False
decimal_places = 1 # number of decimal places to round to

sub_set = None
sub_set_name = ''

vipbench_metadata = os.path.join(vipbench_path, "vip-bench-meta-data.json")
with open(vipbench_metadata, 'r') as f:
    data = json.load(f)

counter = Counter()
cap_set_list = []
cap_set_counter = []
len_data = 0
for id, value in data.items():
    if sub_set is not None and id not in sub_set:
        continue
    question = value["question"]
    answer = value["answer"]
    cap = value["capability"]
    cap = set(cap)
    counter.update(cap)
    if cap not in cap_set_list:
        cap_set_list.append(cap)
        cap_set_counter.append(1)
    else:
        cap_set_counter[cap_set_list.index(cap)] += 1
    
    len_data += 1

sorted_list = counter.most_common()
columns = [k for k, v in sorted_list]
columns.append("total")
df = pd.DataFrame(columns=columns)

for results_file_idx, results_file in enumerate(results_files):
    model_results_file = os.path.join(result_path, results_file)
    if results_file_idx == 0:
        results_str = f"B-Lxx-Mx.x"
    else:
        results_str = f"S-L{str(int(results_file.split('_')[1].split('=')[-1])).zfill(2)}-M{str(float(results_file.split('_')[2].split('=')[-1]))}"
    
    cap_set_sorted_indices = np.argsort(-np.array(cap_set_counter))
    new_cap_set_list = []
    new_cap_set_counter = []
    for index in cap_set_sorted_indices:
        new_cap_set_list.append(cap_set_list[index])
        new_cap_set_counter.append(cap_set_counter[index])

    cap_set_list = new_cap_set_list
    cap_set_counter = new_cap_set_counter
    cap_set_names = ["_".join(list(cap_set)) for cap_set in cap_set_list]

    columns2 = cap_set_names
    columns2.append("total")
    df2 = pd.DataFrame(columns=columns2)

    # grade results for each sample to svae
    grade_file = f"{model}-{vipbench_split}_{gpt_model.replace('/', '_')}-grade-{num_run}runs.json"
    grade_file = os.path.join(grade_path, grade_file)

    # score results regarding capabilities/capability integration to save
    cap_score_file = f"{model}-{vipbench_split}_{sub_set_name}{gpt_model.replace('/', '_').lower()}-cap-score-{num_run}runs.csv"
    cap_score_file = os.path.join(grade_path, cap_score_file)
    cap_int_score_file = f"{model}-{vipbench_split}_{sub_set_name}{gpt_model.replace('/', '_')}-cap-int-score-{num_run}runs.csv"
    cap_int_score_file = os.path.join(grade_path, cap_int_score_file)

    with open(model_results_file) as f:
        try:
            results = json.load(f)
        except Exception:
            continue
    
    grade_results = {}

    def need_more_runs():
        need_more_runs = False
        if len(grade_results) > 0:
            for k, v in grade_results.items():
                if len(v['score']) < num_run:
                    need_more_runs = True
                    break
        return need_more_runs or len(grade_results) < len_data

    questions_json_file = os.path.join(vipbench_path, vipbench_split, "vip_image_oe_qa.jsonl")
    questions= {}
    with open(questions_json_file, 'r') as f:
        for line in f.readlines():
            tmp_data =  json.loads(line)
            tmp_data =  json.loads(line)
            question_id = tmp_data['question_id']
            questions[f'v1_{question_id}'] = tmp_data['text']
        
    while need_more_runs():
        for j in range(num_run):
            print(f'eval run {j}, {results_file}')
            pbar = tqdm(total=len(data.items()), desc="Processing", leave=False)
            for item_id, (id, line) in enumerate(data.items()):
                if sub_set is not None and id not in sub_set:
                    continue
                if id in grade_results and len(grade_results[id]['score']) >= (j + 1):
                    continue
                try:
                    model_pred = results[item_id]['model_output']
                except Exception:
                    break
                # model_pred = results[id]
                question = prompt + '\n' + ' | '.join([questions[id], line['answer'].replace("<AND>", " <AND> ").replace("<OR>", " <OR> "), model_pred, ""])
                messages = [
                    {"role": "user", "content": question},
                ]

                if id not in grade_results:
                    sample_grade = {'model': [], 'content': [], 'score': []}
                else:
                    sample_grade = grade_results[id]

                
                grade_sample_run_complete = False
                temperature = 0.0

                while not grade_sample_run_complete:
                    completion = client.chat.completions.create(
                        model=gpt_model,
                        max_tokens=3,
                        temperature=temperature,
                        messages=messages
                    )
                    content = completion.choices[0].message.content
                    try:
                        score = float(content)
                        if score > 1.0 or score < 0.0:
                            score = 0.0
                    except Exception:
                        score = 0.0
                    flag = True
                    try_time = 1
                    grade_sample_run_complete = True

                if len(sample_grade['model']) >= j + 1:
                    sample_grade['model'][j] = completion.model
                    sample_grade['content'][j] = content
                    sample_grade['score'][j] = score
                else:
                    sample_grade['model'].append(completion.model)
                    sample_grade['content'].append(content)
                    sample_grade['score'].append(score)
                grade_results[id] = sample_grade

                with open(grade_file, 'w') as f:
                    json.dump(grade_results, f, indent=4)
                pbar.update(1)
                    
    assert not need_more_runs()
    cap_socres = {k: [0.0]*num_run for k in columns}
    counter['total'] = len_data

    cap_socres2 = {k: [0.0]*num_run for k in columns2}
    counter2 = {columns2[i]:cap_set_counter[i] for i in range(len(cap_set_counter))}
    counter2['total'] = len_data

    for k, v in grade_results.items():
        if sub_set is not None and k not in sub_set:
            continue
        for i in range(num_run):
            score = v['score'][i]
            caps = set(data[k]['capability'])
            for c in caps:
                cap_socres[c][i] += score
            
            cap_socres['total'][i] += score

            index = cap_set_list.index(caps)
            cap_socres2[cap_set_names[index]][i] += score
            cap_socres2['total'][i] += score

    for k, v in cap_socres.items():
        cap_socres[k] = np.array(v) / counter[k] *100

    std = round(cap_socres['total'].std(), decimal_places)
    total_copy = cap_socres['total'].copy()
    runs = str(list(np.round(total_copy, decimal_places)))

    for k, v in cap_socres.items():
        cap_socres[k] = round(v.mean(), decimal_places)

    df.loc[results_str] = cap_socres

    # print(cap_socres)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)   

df.to_excel(f'results/refer/{args.model_name}-{args.model_size}xvip_image_oe_qa.xlsx')



