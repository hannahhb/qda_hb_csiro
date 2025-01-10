import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, set_seed
import pandas as pd
import random
import re
from tqdm import tqdm
import os
file_path = "data/data1.csv"
# Extract samples for the DOMAIN "Innovation"
from utils.data_preprocessing import *
from utils.config import * 

os.environ["HF_HOME"] = "/scratch3/ban146/.cache"


train_comments, train_constructs, test_comments, test_constructs = extract_stratified_samples(file_path, DOMAIN, 15)


construct_descriptions, _ = extract_construct_descriptions(DOMAIN, CFIR_CONSTRUCTS_FILE)
prompt_intro = f"You are a qualitative data analyser. You are given the following constructs: \n"
main_question = "You need to output the constructs found in the above data. \n"
construct_descriptions = "\n".join(construct_descriptions)
# Create the messages list

# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "meta-llama/Llama-3.1-8B-Instruct"
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
# model_id = "BioMistral/BioMistral-7B"

model = AutoModelForCausalLM.from_pretrained(
                                            model_id,
                                            torch_dtype=torch.bfloat16,
                                            device_map="balanced") #.to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token


results = []

for idx, comment in enumerate(tqdm(test_comments)):

    messages = [
        {"role": "system", "content": prompt_intro + construct_descriptions}
    ]

    for comment_, construct in zip(train_comments, train_constructs):
        messages.append({"role": "user", "content": f"Data: {comment_} \n {main_question}" })
        messages.append({"role": "assistant", "content": f"Output: {construct} "}) #\n Explanation: {explanations[explanation_id]}
        # explanation_id+=1

    messages.append({"role": "user", "content": f"Data: {comment} \n {main_question} Think step by step about why you might assign a construct to a comment. No need to output the explanation."})

    inputs = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        # add_generation_prompt=True,
        return_tensors="pt").to(model.device)

    print(f"Input: {comment}")

    unique_constructs = CFIR_CONSTRUCTS_FILE.dropna(subset=['Construct'])
    unique_constructs = unique_constructs['Construct'].unique()

    # Initialize variables to store outputs and count occurrences
    outputs_list = []
    # construct_count = {construct: 0 for construct in innovation_constructs}
    constructs = []
    # total_samples = 1


    # # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            **GENERATION_ARGS,
        )


    # Decode the response
    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)

    print(f"Processing comment {idx+1}/{len(test_comments)}")
    print(f"{response}\n")
    print(f"Actual output: {test_constructs[idx]}")
    # Store the response
    outputs_list.append(response)

    # Check if each construct is in the response
    for construct in unique_constructs:
        if re.search(rf"\b{re.escape(construct)}\b", response, re.IGNORECASE):
            # construct_count[construct] += 1
            constructs.append(construct)
    result = {
        "comment": comment,
        "predicted_constructs": constructs,
        "actual_constructs": test_constructs[idx],
        "model_response": response
    }
    results.append(result)
    # predicted_constructs_list.append(constructs)

    # test
    # ground_truth_constructs_list.append([test_constructs[idx]])

from utils.evaluation import *

OUTPUT = "predicted_constructs_output.json"
evaluate_and_log(results, OUTPUT)
