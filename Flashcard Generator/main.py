from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from app.scripts.data_extraction import TermDefinitionExtractor
from app.scripts.flashcard_generation import FlashcardGeneration
from app.scripts.dataset_preprocessing import preprocess_function, tokenizer, config
from app.scripts.model_training import train_model
from datasets import load_dataset, load_from_disk
import nltk
import yaml

app = FastAPI()

# Load configuration
current_path = Path(__file__).resolve().parent
config_file_path = current_path / "config.yaml"

with open(config_file_path, 'r') as f:
    config = yaml.safe_load(f)

# Ensure the punkt tokenizer is downloaded
nltk.download('punkt')

class Document(BaseModel):
    content: str

@app.post("/extract_terms/")
def extract_terms(document: Document):
    extractor = TermDefinitionExtractor()
    term_definition_dict = extractor.get_term_definition_dict(document.content)
    if not term_definition_dict:
        raise HTTPException(status_code=404, detail="No terms found.")
    return term_definition_dict

@app.post("/generate_flashcards/")
def generate_flashcards(document: Document):
    flashcard_generator = FlashcardGeneration()
    flashcards_dict = flashcard_generator.generate_flashcards_dict(document.content)
    if not flashcards_dict:
        raise HTTPException(status_code=404, detail="No flashcards generated.")
    return flashcards_dict

@app.post("/preprocess_wikipedia/")
def preprocess_wikipedia():
    # Preprocessing and tokenizing the dataset
    sample_size = 100000  # number of samples to load
    wikipedia_dataset = load_dataset("wikipedia", "20220301.en", split=f"train[:{sample_size}]", trust_remote_code=True)
    
    train_test_ratio = 0.8
    train_test_split = wikipedia_dataset.train_test_split(test_size=1 - train_test_ratio)
    raw_train_dataset = train_test_split['train']
    raw_test_dataset = train_test_split['test']

    tokenized_train_dataset = raw_train_dataset.map(preprocess_function, batched=True, num_proc=4)
    tokenized_test_dataset = raw_test_dataset.map(preprocess_function, batched=True, num_proc=4)

    # Save tokenized datasets
    tokenized_train_dataset.save_to_disk(config['output']['tokenized_train_dataset'])
    tokenized_test_dataset.save_to_disk(config['output']['tokenized_test_dataset'])

    return {"message": "Wikipedia dataset has been preprocessed and saved."}

@app.post("/train_model/")
def train_model_endpoint():
    train_result = train_model()
    return train_result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
