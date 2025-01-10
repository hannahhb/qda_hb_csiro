import dspy
from vllm import LLM
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, set_seed
import torch 
import dspy
from dspy.evaluate import Evaluate
from tqdm import tqdm 
from utils.data_preprocessing import * 
from utils.config import *
from dspy_signature import *
from utils.evaluation import *

import csv
import os 
from typing import Literal
import numpy as np 
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.preprocessing import MultiLabelBinarizer


# os.environ["HF_HOME"] = "/scratch3/ban146/.cache"
# os.environ["HF_TOKEN"] = "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"

lm = dspy.LM("openai/mistralai/Mistral-7B-Instruct-v0.3",
             api_base="http://localhost:7501/v1",  # ensure this points to your port
             api_key="local", model_type='chat', temperature = 0.3)
dspy.configure(lm=lm)

# comment = "Project officers from Divisions where a provider was not interviewed indicated that there was low, if any, uptake of the T-CBT pilot. Low uptake was reported to be associated with the types of challenges mentioned above and others including: most clients opting to have face-to-face sessions even if required to travel long distances, staff turnover (of project officers and T-CBT-trained   mental   health   professionals), Division management issues, the need for increased marketing, and the lack of availability of further training for new mental health professionals."
# # Here is how we call this module
# classification = classify(comment=comment)
# print(classification)

###############################################################################
# 3) Load and filter CSV data by domain (just like before)
###############################################################################
file_path = "data/data1.csv"
# We'll store a dict: comment_text -> set of constructs
comment_to_constructs = {}

with open(file_path, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row.get('Domain', '').strip() == DOMAIN:
            # 'Comments' is your text, 'CFIR Construct 1' is the label
            comment_text = row['Comments']
            cfir_label = row['CFIR Construct 1']

            if comment_text not in comment_to_constructs:
                comment_to_constructs[comment_text] = set()

            comment_to_constructs[comment_text].add(cfir_label)


trainset = []
for comment_text, construct_set in comment_to_constructs.items():
    construct_list = list(construct_set)
    
    ex = dspy.Example(
        comments=comment_text,
        cfir_context="\n".join(descriptions),        # Provide the CFIR context here
        cfir_construct=construct_list
    ).with_inputs("comments", "cfir_context")  # <--- Must list both input fields

    trainset.append(ex)

    
# print(trainset[0])

# OPTIMISED_PATH = "mistrla_optimised_dspy_v2.json"
# tp = dspy.MIPROv2(metric=validate_construct, auto="medium", num_threads=24)
# optimized = tp.compile(classify, trainset=trainset, max_bootstrapped_demos=10, max_labeled_demos=15)
# optimized.save(OPTIMISED_PATH)
# optimised = tp()
optimised = classify 
optimised.load("mistral_optimised_dspy.json")

evaluator = Evaluate(
    devset=trainset,
    num_threads=1,
    display_progress=True,
    display_table=5,
)

# Assume `classify` is your dspy pipeline that outputs `.cfir_construct`.
# results = evaluator(
#     classify,
#     metric=validate_construct,
#     # aggregator=cohen_kappa_aggregator
#     # return_traces=True
# )

import csv
import dspy
from typing import Literal
from sklearn.metrics import precision_recall_fscore_support


results = []
for ex in tqdm(trainset):
    # ex.inputs() gives {"comments": "...", "cfir_context": "..."} if relevant
    prediction = optimised(**ex.inputs())  # your pipeline call
    result = {
        "comment": ex.comments,
        "predicted_constructs": prediction.cfir_construct,
        "actual_constructs": ex.cfir_construct,
        "model_response": None
    }
    # Both `ex` and `prediction` have a .cfir_construct list for multi-label
    results.append(result)

from utils.evaluation import *

OUTPUT = "predicted_constructs_output.json"
evaluate_and_log(results, OUTPUT)
