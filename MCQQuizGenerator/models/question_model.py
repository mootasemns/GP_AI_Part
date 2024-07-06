import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class QuestionModel:
    def __init__(self, model_name, device):
        self.device = device
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def generate_question(self, context, answer, config):
        text = "context: {} answer: {}".format(context, answer)
        encoding = self.tokenizer.encode_plus(text, max_length=config['question_generator']['max_len'], pad_to_max_length=False, truncation=True, return_tensors="pt").to(self.device)
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        with torch.no_grad():
            outs = self.model.generate(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       early_stopping=True,
                                       num_beams=config['question_generator']['num_beams'],
                                       num_return_sequences=1,
                                       no_repeat_ngram_size=2,
                                       max_length=config['question_generator']['max_length'])

        dec = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        question = dec[0].replace("question:", "").strip()

        return question
