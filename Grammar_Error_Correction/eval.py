from transformers import T5Tokenizer, T5ForConditionalGeneration
import json


# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)
    
    
# Load the trained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(config['model_path'])
tokenizer = T5Tokenizer.from_pretrained(config['tokenizer_path'])

def correct_grammar(sentence):
    # Prefixing the input text with 'grammar:' is optional and depends on how the model was trained.
    # If you used a specific prefix during training (like "grammar correction:"), use the same here.
    inputs = tokenizer.encode("grammar: " + sentence, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate output using the model
    outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    
    # Decode the generated ids to a string
    corrected_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_sentence

# # Test the function
# sentence = "i was watched tv then my father has call me for do her work"
# corrected_sentence = correct_grammar(sentence)
# print("Original:", sentence)
# print("Corrected:", corrected_sentence)