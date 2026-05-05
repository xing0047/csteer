import json
import argparse
import os
import pandas as pd
import numpy as np

from tqdm import tqdm
from openai import OpenAI
from collections import Counter
from behaviors import (
    get_inst_it_image_mc_data,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="internvl3_5", choices=["internvl3_5"])
parser.add_argument("--model_size", type=str, default="8b", choices=["8b"])
parser.add_argument("--layers", nargs="+", type=int, required=True)

args = parser.parse_args()

###### change your model name ######
model = f"{args.model_name}_{args.model_size}"
gpt_model = "Qwen/Qwen2.5-72B-Instruct"
type = "inst_it_image_mc_qa"
verbose = True
result_path = os.path.join("results", "refer")
grade_path = os.path.join("grades", "refer")
num_run = 1 # we set it as 5 in the paper

results_files = [f"results_layer=00_multiplier=0.0_behavior=refer_type={type}_model_name={args.model_name}_model_size={args.model_size}.json"]
result_layers = ['xx']
result_multipliers = ['x.x']
for layer in args.layers:
    for multiplier in [1.0]:
        result_layers.append(str(layer).zfill(2))
        result_multipliers.append(multiplier)
        results_files.append(f"results_layer={str(layer).zfill(2)}_multiplier={multiplier}_behavior=refer_type={type}_model_name={args.model_name}_model_size={args.model_size}.json")

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-123456",
)

prompt = """Compare the Prediction from AI models and Ground Truth, to give a correctness score for the prediction (A, B, C, D). The correctness score is either 0.0 if Prediction is wrong, or 1.0 if Prediction is correct. Three examples are as below,

## Prediction: In the image, <object9> is a person wearing sunglasses and a hat. The person is not wearing a white shoe. So, the correct answer is (C) A white shoe.
## Ground Truth: (C) A white shoe
## Score: 1.0

## Prediction: In the image, <object9> is a person wearing sunglasses and a watch. The person is not wearing a hat. So, the correct answer is (B) A hat.
## Ground Truth: (C) A white shoe
## Score: 0.0

## Prediction: In the provided video, we see a person flying a small aircraft, wearing a flight harness with blue straps. The straps stretch and extend across various parts of the seat and the aircraft's interior, indicating their role in securing the person during flight.
The highlighted objects are marked in two colors: red for <object2> and green for <object3>. Since both are straps of the same color (red and blue), it is clear that they are part of the same set and serve the same functional purpose within the context of the flight.
Given this information:
A) <object2> is held by <object3> - This is incorrect. The straps are part of the seating system, not held by anyone or anything else that can be shown from this angle.
B) <object2> is next to <object3> - This is incorrect. The straps are positioned near each other but are not shown to be adjacent to each other from the given angle.
C) <object2> is above <object3> - This is incorrect. The straps are not above or below each other as they are attached to the seating surface.
D) <object2> is behind <object3> - This is incorrect. The straps in the image are not behind each other but are part of the same structure, stretching out across various parts of the seat.
Therefore, the correct relationship is:
(A) <object2> is held by <object3> - This is not accurate as the straps do not represent anyone or an attachment mechanism.
Considering the positioning and nature of the straps within the image, the correct answer is that there is not a clear relationship in the context provided by the color masks. The red and blue straps are part of the same set used for securing a harness in the aircraft.
## Ground Truth: (A) <object2> is held by <object3>
## Score: 1.0

For the pair, 
## Prediction: {}
## Ground Truth: {}
## Score: 
output the correctness score ONLY.
"""

# load metadata
use_sub_set = False
decimal_places = 1 # number of decimal places to round to

sub_set = None
sub_set_name = ''

data = get_inst_it_image_mc_data()

counter = Counter()
cap_set_list = []
cap_set_counter = []
len_data = 0
for value in data:
    if sub_set is not None and id not in sub_set:
        continue
    question = value["Question"]
    answer = value["Answer"]
    cap = value["type"]
    cap = cap[:3].lower()
    counter.update([cap])
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

for results_file_idx, (result_layer, result_multiplier, results_file) in enumerate(zip(result_layers, result_multipliers, results_files)):
    model_results_file = os.path.join(result_path, results_file)
    if results_file_idx == 0:
        results_str = f"B-Lxx-Mx.x"
    else:
        results_str = f"S-L{str(result_layer).zfill(2)}-M{str(float(results_file.split('_')[2].split('=')[-1]))}"
    
    cap_set_sorted_indices = np.argsort(-np.array(cap_set_counter))
    new_cap_set_list = []
    new_cap_set_counter = []
    for index in cap_set_sorted_indices:
        new_cap_set_list.append(cap_set_list[index])
        new_cap_set_counter.append(cap_set_counter[index])

    cap_set_list = new_cap_set_list
    cap_set_counter = new_cap_set_counter
    cap_set_names = ["_".join(list(cap_set)) for cap_set in cap_set_list]

    # columns2 = cap_set_names
    # columns2.append("total")
    # df2 = pd.DataFrame(columns=columns2)

    # grade results for each sample to svae
    grade_file = f"{model}-{gpt_model.replace('/', '_')}-{type}-layer{str(result_layer).zfill(2)}-multiplier{str(result_multiplier)}-grade-{num_run}runs.json"
    grade_file = os.path.join(grade_path, grade_file)

    # score results regarding capabilities/capability integration to save
    cap_score_file = f"{model}-{sub_set_name}{gpt_model.replace('/', '_').lower()}-cap-score-{num_run}runs.csv"
    cap_score_file = os.path.join(grade_path, cap_score_file)
    cap_int_score_file = f"{model}-{sub_set_name}{gpt_model.replace('/', '_')}-cap-int-score-{num_run}runs.csv"
    cap_int_score_file = os.path.join(grade_path, cap_int_score_file)

    with open(model_results_file) as f:
        try:
            results = json.load(f)
        except Exception:
            continue
    
    if not os.path.exists(grade_file):
        grade_results = {}

        def need_more_runs():
            need_more_runs = False
            if len(grade_results) > 0:
                for k, v in grade_results.items():
                    if len(v['score']) < num_run:
                        need_more_runs = True
                        break
            return need_more_runs or len(grade_results) < len_data

        while need_more_runs():
            for j in range(num_run):
                print(f'eval run {j}, {results_file}')
                pbar = tqdm(total=len(data), desc="Processing", leave=False)
                for item_id, line in enumerate(data):
                    if sub_set is not None and item_id not in sub_set:
                        continue
                    if item_id in grade_results and len(grade_results[item_id]['score']) >= (j + 1):
                        continue
                    try:
                        model_pred = results[item_id]['model_output']
                    except Exception:
                        break
                    # model_pred = results[id]
                    question = prompt.format(model_pred, line['Answer'])
                    messages = [
                        {"role": "user", "content": question},
                    ]

                    if item_id not in grade_results:
                        sample_grade = {'model': [], 'content': [], 'score': []}
                    else:
                        sample_grade = grade_results[item_id]
                    
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
                        if verbose:
                            print(f'P: {model_pred}')
                            print(f"G: {line['Answer']}")
                            print(f"C: {content}")
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
                    grade_results[item_id] = sample_grade

                    with open(grade_file, 'w') as f:
                        json.dump(grade_results, f, indent=4)
                    pbar.update(1)
    else:
        grade_results = json.load(open(grade_file))
                    
    # assert not need_more_runs()
    cap_socres = {k: [0.0]*num_run for k in columns}
    counter['total'] = len_data

    # cap_socres2 = {k: [0.0]*num_run for k in columns2}
    # counter2 = {columns2[i]:cap_set_counter[i] for i in range(len(cap_set_counter))}
    # counter2['total'] = len_data

    for k, v in grade_results.items():
        if sub_set is not None and k not in sub_set:
            continue
        for i in range(num_run):
            score = v['score'][i]
            cap = data[int(k)]['type']
            cap = cap[:3].lower()
            cap_socres[cap][i] += score
            
            cap_socres['total'][i] += score

            # index = cap_set_list.index(caps)
            # cap_socres2[cap_set_names[index]][i] += score
            # cap_socres2['total'][i] += score

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

df.to_excel(f'results/refer/{args.model_name}-{args.model_size}x{type}.xlsx')



