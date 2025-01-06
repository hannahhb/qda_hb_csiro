import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, set_seed
import pandas as pd
import random
import re
from tqdm import tqdm

file_path = "data/data1.csv"
# Extract samples for the DOMAIN "Innovation"

DOMAIN = "Innovation"

def extract_construct_descriptions(domain_name, df):
    # Filter rows that belong to the requested DOMAIN
    filtered_df = df[df['Domain'].str.contains(domain_name, case=False, na=False)]

    # Extract the descriptions from 'Unnamed: 1' and 'Unnamed: 2'
    descriptions = []
    for _, row in filtered_df.iterrows():
        CFIR = row['Construct']
        description = row['Description']
        if pd.notna(CFIR) and pd.notna(description):
            descriptions.append(f"{CFIR}: {description}")

    # Format and return the descriptions
    return "\n".join(descriptions)

# Helper function to get all constructs associated with given comments
def get_constructs_for_comments(df, comments):
    return [df.loc[df["Comments"] == c, 'CFIR Construct 1'].unique().tolist() for c in comments]


def extract_stratified_samples(file_path, domain_name, num_samples=5):
    # Load and filter the CSV file
    df = pd.read_csv(file_path)
    filtered_df = df[df['Domain'].str.contains(domain_name, case=False, na=False)]
    filtered_df = filtered_df.dropna(subset=['Comments', 'CFIR Construct 1']).reset_index(drop=True)

    # Identify unique constructs
    unique_constructs = filtered_df['CFIR Construct 1'].unique()
    num_constructs = len(unique_constructs)

    # Distribute the total number of samples across constructs
    base_samples = num_samples // num_constructs
    extra = num_samples % num_constructs

    samples_per_construct = {c: base_samples for c in unique_constructs}
    random.seed(42)
    constructs_list = list(unique_constructs)
    random.shuffle(constructs_list)
    for i in range(extra):
        samples_per_construct[constructs_list[i]] += 1


    # Sample comments per construct without duplicates
    sampled_comments_set = set()
    sampled_comments = []
    for construct, n_samples in samples_per_construct.items():
        construct_df = filtered_df[(filtered_df['CFIR Construct 1'] == construct) &
                                   (~filtered_df['Comments'].isin(sampled_comments_set))]
        n_samples = min(n_samples, len(construct_df))
        if n_samples > 0:
            chosen_comments = construct_df.sample(n=n_samples, random_state=21)['Comments'].tolist()
            sampled_comments.extend(chosen_comments)
            sampled_comments_set.update(chosen_comments)

    # Prepare train sets
    train_comments = sampled_comments
    train_constructs = get_constructs_for_comments(filtered_df, train_comments)

    # Prepare test sets (comments not in train)
    test_df = filtered_df[~filtered_df['Comments'].isin(sampled_comments_set)].reset_index(drop=True)
    test_comments = test_df['Comments'].unique().tolist()
    test_constructs = get_constructs_for_comments(test_df, test_comments)

    return train_comments, train_constructs, test_comments, test_constructs


train_comments, train_constructs, test_comments, test_constructs = extract_stratified_samples(file_path, DOMAIN, 15)

CFIR_constructs = pd.read_csv("data/CFIR_constructs.csv")

construct_descriptions = extract_construct_descriptions(DOMAIN, CFIR_constructs)
prompt_intro = f"You are a qualitative data analyser. You are given the following constructs: \n"
main_question = "You need to output the constructs found in the above data. \n"

# Create the messages list

predicted_constructs_list = []
ground_truth_constructs_list = []
print(len(test_comments))


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

    data = pd.read_csv("data/CFIR_constructs.csv")

    innovation_constructs = data[data['Domain'] == 'Innovation']
    innovation_constructs = innovation_constructs.dropna(subset=['Construct'])
    innovation_constructs = innovation_constructs['Construct'].unique()

    # Initialize variables to store outputs and count occurrences
    outputs_list = []
    # construct_count = {construct: 0 for construct in innovation_constructs}
    constructs = []
    # total_samples = 1


    generation_args = {
        "max_new_tokens": 1024,
        "temperature": 0.3,
        # "num_beams": 5,
        # "early_stopping": True,
        # "no_repeat_ngram_size": 2,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.92,
        # "num_return_sequences":3,
        # "do_sample": False,
    }

    # # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            **generation_args,
        )


    # Decode the response
    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)

    print(f"Processing comment {idx+1}/{len(test_comments)}")
    print(f"{response}\n")
    print(f"Actual output: {test_constructs[idx]}")
    # Store the response
    outputs_list.append(response)

    # Check if each construct is in the response
    for construct in innovation_constructs:
        if re.search(rf"\b{re.escape(construct)}\b", response, re.IGNORECASE):
            # construct_count[construct] += 1
            constructs.append(construct)

    predicted_constructs_list.append(constructs)

    # test
    # ground_truth_constructs_list.append([ground_truth_constructs[idx]])
