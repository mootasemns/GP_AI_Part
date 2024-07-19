import transformers
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import accelerate
from datasets import load_from_disk
from evaluate import load
import numpy as np
import nltk
import yaml
from pathlib import Path

# Ensure the punkt tokenizer is downloaded
nltk.download('punkt')

current_path = Path(__file__).resolve().parent

config_file_path = current_path.parent / "config.yaml"

with open(config_file_path, 'r') as f:
    config = yaml.safe_load(f)

model_checkpoint = config['model']['checkpoint']
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint)

# Load tokenized datasets
tokenized_train_dataset = load_from_disk(config['output']['tokenized_train_dataset'])
tokenized_valid_dataset = load_from_disk(config['output']['tokenized_valid_dataset'])

metric = load("rouge")

batch_size = 16
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned-wikipedia",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False, 
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save the fine-tuned model
save_directory = config['flashcard_generation']['fine_tuned_model_directory']
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
