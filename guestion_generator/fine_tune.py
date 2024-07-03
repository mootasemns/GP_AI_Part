### Fine-tuned model
import transformers
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_from_disk
from evaluate import load
import numpy as np
import nltk

# download the punkt tokenizer
nltk.download('punkt')

# Specify the model checkpoint to be used
model_checkpoint = "t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint) # Load the T5 model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint) # Load the tokenizer

# Load tokenized datasets
tokenized_train_dataset = load_from_disk("tokenized_train_dataset")
tokenized_valid_dataset = load_from_disk("tokenized_valid_dataset")

# Load the rouge metric for evaluation
metric = load("rouge")

batch_size = 16
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned-squad",  # Directory to save the fine-tuned model
    evaluation_strategy="epoch",  # Evaluate the model at the end of each epoch
    learning_rate=2e-5,  # Learning rate
    per_device_train_batch_size=batch_size,  # Training batch size for 
    per_device_eval_batch_size=batch_size,  # Evaluation batch size for 
    weight_decay=0.01,  # Weight decay for regularization
    save_total_limit=3,  # Limit the total number of saved checkpoints
    num_train_epochs=1,  # Num of training epochs
    predict_with_generate=True,  # Enable generation during evaluation
    fp16=True,  # Use 16-bit (mixed) precision training
    push_to_hub=False,  # Set to False to skip pushing to Hugging Face Hub
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Compute_metrics func
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Split the decoded predictions and labels into sentences for better evaluation
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    # Compute rouge scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    result = {key: value * 100 for key, value in result.items()} # Convert scores to percentages
    
    # Avg lenght of gen pred
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Save the fine-tuned model & tokenizer
save_directory = "./fine_tuned_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
