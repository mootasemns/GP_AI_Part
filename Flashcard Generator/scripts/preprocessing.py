from pathlib import Path
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
import nltk
import yaml

nltk.download('punkt')  # download punkt tokenizer

# Load configuration
current_path = Path(__file__).resolve().parent.parent
config_file_path = current_path / "config.yaml"

with open(config_file_path, 'r') as f:
    config = yaml.safe_load(f)

model_checkpoint = config['model']['checkpoint']  # the model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)  # the tokenizer

# Load a sample of the Wikipedia dataset
sample_size = 100000  # number of samples to load
wikipedia_dataset = load_dataset("wikipedia", "20220301.en", split=f"train[:{sample_size}]", trust_remote_code=True)

# Split the dataset into train and test sets before tokenizing
train_test_ratio = 0.8
train_test_split = wikipedia_dataset.train_test_split(test_size=1 - train_test_ratio)
raw_train_dataset = train_test_split['train']
raw_test_dataset = train_test_split['test']

prefix = "generate question: " if model_checkpoint in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"] else ""

max_input_length = config['preprocessing']['max_input_length']
max_target_length = config['preprocessing']['max_target_length']

def preprocess_function(examples):
    inputs = [prefix + context for context in examples["text"]]  # add the `prefix` to each context in the input `examples`

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)  # set the max\min size to the input # we tokenize the input

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=[context for context in examples["text"]], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize the train and test datasets separately with multiprocessing
tokenized_train_dataset = raw_train_dataset.map(preprocess_function, batched=True, num_proc=4)
tokenized_test_dataset = raw_test_dataset.map(preprocess_function, batched=True, num_proc=4)

# Save tokenized datasets
tokenized_train_dataset.save_to_disk(config['output']['tokenized_train_dataset'])
tokenized_test_dataset.save_to_disk(config['output']['tokenized_test_dataset'])
