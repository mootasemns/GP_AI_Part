import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class SummaryModel:
    def __init__(self, model_name, device):
        self.device = device
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def summarize(self, text, config):
        text = text.strip().replace("\n", " ")
        text = "summarize: " + text
        max_len = config['summarizer']['max_len']

        encoding = self.tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=False, truncation=True, return_tensors="pt")
        input_ids, attention_mask = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outs = self.model.generate(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       early_stopping=True,
                                       num_beams=config['summarizer']['num_beams'],
                                       num_return_sequences=1,
                                       no_repeat_ngram_size=2,
                                       min_length=config['summarizer']['min_length'],
                                       max_length=config['summarizer']['max_length'])

        dec = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        summary = dec[0]

        return summary
