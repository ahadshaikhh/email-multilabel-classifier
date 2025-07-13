import pandas as pd

def load_data(file_path):
    """
    Loads the CSV file and returns a clean DataFrame with required columns only.
    Drops missing values and renames 'Interaction content' to 'Text'.
    """
    df = pd.read_csv(file_path)
    df = df[['Interaction content', 'Type 2', 'Type 3', 'Type 4']].dropna()
    df = df.rename(columns={'Interaction content': 'Text'})
    return df
