import pandas as pd
import re
from Config import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def get_input_data() -> pd.DataFrame:
    df1 = pd.read_csv("/Users/ahadshaikh/Desktop/CA_EEAS/email-multilabel-classifier/data/AppGallery.csv", skipinitialspace=True)
    df1.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)

    df2 = pd.read_csv("data/Purchasing.csv", skipinitialspace=True)
    df2.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)

    df = pd.concat([df1, df2])
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    df["y"] = df[Config.CLASS_COL]
    df = df.loc[(df["y"] != '') & (~df["y"].isna())]
    return df

def clean_text(text: str) -> str:
    text = text.lower()
    return re.sub(r'[^a-zA-Z0-9 ]', '', text)

def preprocess_data(df: pd.DataFrame, text_column: str = 'Interaction content'):
    # Ensure column names are clean
    df.columns = df.columns.str.strip()

    # Check for required label columns
    required_cols = ['Type 2', 'Type 3', 'Type 4']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    # Get input texts
    texts = df[text_column].fillna("")

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(texts)

    # Label Encoding
    le2 = LabelEncoder()
    le3 = LabelEncoder()
    le4 = LabelEncoder()

    y2 = le2.fit_transform(df['Type 2'].astype(str))
    y3 = le3.fit_transform(df['Type 3'].astype(str))
    y4 = le4.fit_transform(df['Type 4'].astype(str))

    return X_vec, y2, y3, y4, vectorizer, le2, le3, le4
