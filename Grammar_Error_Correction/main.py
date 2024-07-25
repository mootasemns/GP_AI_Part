from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from eval import correct_grammar

app = FastAPI()

class TextData(BaseModel):
    text: str

@app.post("/correct_grammar/")
def correct_grammar_endpoint(text_data: TextData):
    try:
        corrected_text = correct_grammar(text_data.text)
        return {"corrected_text": corrected_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Grammar Correction API!"}
