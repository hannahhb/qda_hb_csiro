import os
import pandas as pd 

# File paths
DATA_FILE = "data/data1.csv"
CFIR_CONSTRUCTS_FILE = pd.read_csv("data/CFIR_constructs.csv")
OUTPUT_FILE = "predicted_constructs_output.json"
DOMAIN = "Innovation"

# Model configuration
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_HOME = os.getenv("HF_HOME", "/scratch3/ban146/.cache/")

# Generation arguments
GENERATION_ARGS = {
    "max_new_tokens": 1024,
    "temperature": 0.3,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.92,
}

# W&B configuration
WANDB_PROJECT = "qualitative-data-analysis"
WANDB_RUN_NAME = "automated-prompt-tuning"
