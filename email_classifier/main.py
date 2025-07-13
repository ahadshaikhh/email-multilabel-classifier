from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
from email_classifier.data_loader import load_data
from email_classifier.preprocess import preprocess_data
from email_classifier.model.chained import ChainedModel

import random
seed =0
random.seed(seed)
np.random.seed(seed)

def load_data():
    #load the input data
    df = get_input_data()
    return  df

def preprocess_data(df):
    # De-duplicate input data
    df =  de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)

if __name__ == '__main__':
    # Load and preprocess data
    df = load_data("data/AppGallery.csv")
    X_vec, y2, y3, y4, vectorizer = preprocess_data(df)

    # Train chained model
    model = ChainedModel()
    model.fit(X_vec, y2, y3, y4)

    # Predict
    y2_pred, y3_pred, y4_pred = model.predict(X_vec)

    # Save models
    model.save_models()
