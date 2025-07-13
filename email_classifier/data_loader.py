import pandas as pd

def load_data(file_path):
    df = pd.read_csv("email-multilabel-classifier/data/AppGallery.csv")
    # Drop Type 1, keep only relevant columns
    df = df[['Text', 'Type 2', 'Type 3', 'Type 4']].dropna()
    return df
