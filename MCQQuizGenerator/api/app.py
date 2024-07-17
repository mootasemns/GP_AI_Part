from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
from pathlib import Path
# Add the project root to the Python path
current_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(current_dir))

from utils.utils import load_config, postprocesstext, get_keywords, get_distractors_wordnet, mmr
from models.summary_model import SummaryModel
from models.question_model import QuestionModel
from models.sense2vec_model import Sense2VecModel
from models.sentence_transformer_model import SentenceTransformerModel
import torch
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

class MCQRequest(BaseModel):
    context: str
    method: str

app = FastAPI()
# Setup CORS
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity, but you should restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = load_config()

device = torch.device(config['device'])

summary_model = SummaryModel(config['models']['t5_summary_model'], device)
question_model = QuestionModel(config['models']['t5_question_model'], device)
current_dir = Path(__file__).parent.parent
s3v_old_path = current_dir / "imports"  / config['models']['sense2vec_model']
sense2vec_model = Sense2VecModel(s3v_old_path)
sentence_transformer_model = SentenceTransformerModel(config['models']['sentence_transformer_model'], device)

@app.post("/generate_mcq/")
async def generate_mcq(request: MCQRequest):
    context = request.context
    method = request.method

    summary_text = summary_model.summarize(context, config)
    np = get_keywords(context, summary_text)

    questions = []

    for answer in np:
        ques = question_model.generate_question(summary_text, answer, config)
        correct_answer = answer.capitalize()

        if method == "Wordnet":
            distractors = get_distractors_wordnet(answer)
        elif method == "Sense2Vec":
            distractors = sense2vec_model.get_words(answer.capitalize(), config['distractors']['top_n'], ques)
        else:
            raise HTTPException(status_code=400, detail="Invalid method. Choose either 'Wordnet' or 'Sense2Vec'.")

        distractors = distractors[:3]  # Limit distractors to top 3

        question_obj = {
            "question": ques,
            "correct_answer": correct_answer,
            "distractors": distractors
        }
        questions.append(question_obj)

    return questions


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
