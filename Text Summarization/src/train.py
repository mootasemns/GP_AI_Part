import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from utils import *
import nltk
from datasets import Dataset, load_metric
import yaml


def compute_metrics(eval_pred, tokenizer):
    metric = load_metric('rouge')
    predictions, labels = eval_pred  # Obtaining predictions and true labels
    
    # Decoding predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Obtaining the true labels tokens, while eliminating any possible masked token (i.e., label = -100)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    # Computing rouge score
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}  # Extracting some results

    # Add mean-generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

def train_model(train_ds, test_ds, val_ds, tokenizer, model, training_args):

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
    )
    trainer.train()

def main():
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_path = config['data']['train_path']
    test_path = config['data']['test_path']
    val_path = config['data']['val_path']
    checkpoint = config['model']['checkpoint']
    output_dir = config['model']['output_dir']
    seed = config['training']['seed']
    learning_rate = config['training']['learning_rate']
    per_device_train_batch_size = config['training']['per_device_train_batch_size']
    per_device_eval_batch_size = config['training']['per_device_eval_batch_size']
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    weight_decay = config['training']['weight_decay']
    num_train_epochs = config['training']['num_train_epochs']
    predict_with_generate = config['training']['predict_with_generate']
    fp16 = config['training']['fp16']
    report_to = config['training']['report_to']

    train, test, val = load_data(train_path, test_path, val_path)
    train_ds = preprocess_data(train)
    test_ds = preprocess_data(test)
    val_ds = preprocess_data(val)

    tokenizer, model = setup_model_and_tokenizer(checkpoint)
    tokenizer.save_pretrained(output_dir)

    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        seed=seed,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        save_total_limit=2,
        num_train_epochs=num_train_epochs,
        predict_with_generate=predict_with_generate,
        fp16=fp16,
        report_to=report_to
    )
    print("Starting Training  ... ")
    train_model(train_ds, test_ds, val_ds, tokenizer, model, training_args)

if __name__ == "__main__" : 
    main()