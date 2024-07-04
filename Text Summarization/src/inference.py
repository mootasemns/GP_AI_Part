from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

app = FastAPI()

# Load the summarization pipeline
summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn')

class TextToSummarize(BaseModel):
    text: str

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity, but you should restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/summarize/")
async def summarize_text(text_to_summarize: TextToSummarize):
    try:
        summary = summarizer(text_to_summarize.text, max_length=130, min_length=30, do_sample=False)
        return {"summary": summary[0]['summary_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8006)
