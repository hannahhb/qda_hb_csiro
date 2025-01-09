import pandas as pd
import random

def extract_construct_descriptions(domain_name, df):
    # Filter rows that belong to the requested DOMAIN
    filtered_df = df[df['Domain'].str.contains(domain_name, case=False, na=False)]

    # Extract the descriptions from 'Unnamed: 1' and 'Unnamed: 2'
    descriptions = []
    constructs = []
    for _, row in filtered_df.iterrows():
        CFIR = row['Construct']
        description = row['Description']
        if pd.notna(CFIR) and pd.notna(description):
            descriptions.append(f"{CFIR}: {description}")
            constructs.append(CFIR)
            
    # Format and return the descriptions
    return descriptions, constructs


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
