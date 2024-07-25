from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from process_data import train_dataset, val_dataset
import json

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)
    
    
#Set the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

def tokenize_function(examples):
    model_inputs = tokenizer(examples['Ungrammatical Statement'], max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['Standard English'], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize the data
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set the model
model = T5ForConditionalGeneration.from_pretrained('t5-small')

### Fine-Tune the model
training_args = TrainingArguments(**config['training_args'])

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# Save the trained model and tokenizer
model.save_pretrained(config['model_path'])
tokenizer.save_pretrained(config['tokenizer_path'])

