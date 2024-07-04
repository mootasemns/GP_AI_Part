import re
import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import Dataset

def clean_tags(text):
    clean = re.compile('<.*?>')
    clean = re.sub(clean, '', text)
    clean = '\n'.join([line for line in clean.split('\n') if not re.match('.*:\s*$', line)])
    return clean

def clean_df(df, cols):
    for col in cols:
        df[col] = df[col].fillna('').apply(clean_tags)
    return df


def setup_device():
    if torch.cuda.is_available():
        print("GPU is available. \nUsing GPU")
        return torch.device('cuda')
    else:
        print("GPU is not available. \nUsing CPU")
        return torch.device('cpu')
        


def load_data(train_path, test_path, val_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    val = pd.read_csv(val_path)
    return train, test, val

def preprocess_data(df):
    # Assuming clean_df is imported from utils.py
    df = clean_df(df, ['dialogue', 'summary'])
    return Dataset.from_pandas(df)


def setup_model_and_tokenizer(checkpoint):
    tokenizer = BartTokenizer.from_pretrained(checkpoint)
    model = BartForConditionalGeneration.from_pretrained(checkpoint)
    return tokenizer, model

def preprocess_function(examples, tokenizer):
    inputs = [doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

