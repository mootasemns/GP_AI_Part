# prepare_dataset.py
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
import nltk

nltk.download('punkt') # download punkt tokenizer

model_checkpoint = "t5-small" # the model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint) # the tokenizer

# the dataset
train_dataset = load_dataset("squad", split='train')
valid_dataset = load_dataset("squad", split='validation')

prefix = "generate question: " if model_checkpoint in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"] else ""

max_input_length = 1024
max_target_length = 128

def preprocess_function(examples):
    inputs = [prefix + context for context in examples["context"]] # add the `prefix` to each context in the input `examples`
   
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True) # set the max\min size to the input # we tokenize the input
    
    # Setup the tokenizer for targets # we tokenize the answers to prepare targets for the model
    labels = tokenizer(text_target=[answer['text'][0] for answer in examples["answers"]], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize the datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True)

# Save tokenized datasets
tokenized_train_dataset.save_to_disk("tokenized_train_dataset")
tokenized_valid_dataset.save_to_disk("tokenized_valid_dataset")
