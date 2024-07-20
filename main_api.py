import subprocess
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from pathlib import Path
import sys
import torch
from transformers import pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoProcessor, AutoModel
from scipy.io.wavfile import write
import yaml
import re
from nltk import sent_tokenize
from question_generator.extract import QuestionExtractor

# Add the project root to the Python path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Import utilities and models for MCQ Generator
from MCQQuizGenerator.models.summary_model import SummaryModel
from MCQQuizGenerator.models.question_model import QuestionModel
from MCQQuizGenerator.models.sense2vec_model import Sense2VecModel
from MCQQuizGenerator.models.sentence_transformer_model import SentenceTransformerModel
from utils import load_config, get_keywords, get_distractors_wordnet, slice_array_wave, split_sentences

# Initialize FastAPI app
app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CFG:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SPEAKER = "v2/en_speaker_0"
    MODEL_NAME = 'suno/bark'
    AMPLITUDE_THRESHOLD = 0.05
    TIME_THRESHOLD = int(24_000 * 0.5)
    IGNORE_INITIAL_STEPS = int(24_000 * 0.5)

# Load config
config = load_config()

# MCQ Generator models
summary_model = SummaryModel(config['models']['t5_summary_model'], device=CFG.DEVICE)
question_model = QuestionModel(config['models']['t5_question_model'], device=CFG.DEVICE)
s3v_old_path = current_dir / "MCQQuizGenerator" / "imports" / config['models']['sense2vec_model']
sense2vec_model = Sense2VecModel(s3v_old_path)
sentence_transformer_model = SentenceTransformerModel(config['models']['sentence_transformer_model'], device=CFG.DEVICE)

# Text Summarization model initialization
inference_config = config["inference"]
model_path = Path(__file__).parent / "MCQQuizGenerator" / "imports" / inference_config["model"]
summarizer = pipeline(inference_config["task"], model=str(model_path), device=CFG.DEVICE)

# Audio processor and model initialization
processor = AutoProcessor.from_pretrained(CFG.MODEL_NAME, voice_preset=CFG.SPEAKER, return_tensors='pt')
audio_model = AutoModel.from_pretrained(CFG.MODEL_NAME, torch_dtype=torch.float16).to(CFG.DEVICE)
audio_model.enable_cpu_offload()
audio_model.eval()
print("Processor and model loaded successfully.")

# Question Generation model
class QuestionGeneration:
    '''This class contains the method to generate questions'''

    def __init__(self):
        current_path = Path(__file__).parent
        config_file_path = current_path / "config" / "models_config.yaml"
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)

        self.num_questions = config['question_generator']['num_questions']
        self.num_options = config['question_generator']['num_options']
        fine_tuned_model_directory = config['question_generator']['fine_tuned_model_directory']

        self.model = T5ForConditionalGeneration.from_pretrained(fine_tuned_model_directory).to(CFG.DEVICE)
        self.tokenizer = T5Tokenizer.from_pretrained(fine_tuned_model_directory)
        self.extractor = QuestionExtractor()

    def clean_text(self, text):
        text = text.replace('\n', ' ')  # remove newline chars
        sentences = sent_tokenize(text)
        cleaned_text = ""
        for sentence in sentences:
            cleaned_sentence = re.sub(r'([^\s\w]|_)+', '', sentence)
            cleaned_sentence = re.sub(' +', ' ', cleaned_sentence)
            cleaned_text += cleaned_sentence

            if cleaned_text[-1] == ' ':
                cleaned_text = cleaned_text[:-1] + '.'
            else:
                cleaned_text += '.'

            cleaned_text += ' '  # pad with a space at the end
        return cleaned_text

    def generate_questions_dict(self, document):
        document = self.clean_text(document)
        input_ids = self.tokenizer(document, return_tensors="pt", max_length=512, truncation=True)['input_ids'].to(CFG.DEVICE)

        self.questions_dict = {}
        for i in range(self.num_questions):
            output = self.model.generate(input_ids, max_length=100, num_return_sequences=1)[0]
            question = self.tokenizer.decode(output, skip_special_tokens=True)
            self.questions_dict[i + 1] = {
                "question": f"What is the significance of '{question}' in the given context?",
                "answer": self.extract_answer(question, document)
            }

        return self.questions_dict

    def extract_answer(self, question, document):
        sentences = sent_tokenize(document)
        for sentence in sentences:
            if question.lower() in sentence.lower():
                return sentence.strip()

        return "Answer not found"

# Request and response models
class MCQRequest(BaseModel):
    context: str
    method: str

class TextToSummarize(BaseModel):
    text: str

class QuestionGenerationRequest(BaseModel):
    document: str

class TextRequest(BaseModel):
    text: str

# API endpoints
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

@app.post("/TextToSpeech/")
async def synthesize_text(text_request: TextRequest):
    try:
        print("Received request with text:", text_request.text)
        text_to_infer = text_request.text

        sentences, number_of_sentences = split_sentences(text=text_to_infer)
        print(f"Split text into {number_of_sentences} sentences.")

        all_audio_arrays = []

        for sentence_number in range(number_of_sentences):
            current_sentence = sentences[sentence_number]
            print(f"Processing sentence {sentence_number + 1}: {current_sentence}")

            inputs = processor(
                text=current_sentence,
                return_tensors="pt",
                return_attention_mask=True,
                max_length=1024,
                voice_preset=CFG.SPEAKER,
                add_special_tokens=False,
            ).to(CFG.DEVICE)

            with torch.inference_mode():
                result = audio_model.generate(
                    **inputs,
                    do_sample=True,
                    semantic_max_new_tokens=1024,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )

            result_array = result.cpu().numpy()
            sample_rate = audio_model.generation_config.sample_rate  # 24_000
            print(f"Generated audio for sentence {sentence_number + 1}")

            all_audio_arrays.append(result_array)

        concatenated_array = np.array([])

        for audio_number, sentence_audio in enumerate(all_audio_arrays):
            print(f'Audio {audio_number + 1}/{len(all_audio_arrays)}')

            # Concatenate audio arrays
            current_array = all_audio_arrays[audio_number].squeeze()

            # Post-process array (remove padding in inference was done in batches)
            current_array = slice_array_wave(
                input_array=current_array,
                amplitude_threshold=CFG.AMPLITUDE_THRESHOLD,
                time_threshold=CFG.TIME_THRESHOLD,
                ignore_initial_steps=CFG.IGNORE_INITIAL_STEPS
            )

            concatenated_array = np.concatenate([concatenated_array, current_array])

        # Generate a unique filename using UUID
        unique_filename = str(uuid.uuid4())
        final_wav_path = f"saved_audio/{unique_filename}.wav"
        mp3_path = f"saved_audio/{unique_filename}.mp3"

        # Save concatenated audio as a WAV file
        write(
            final_wav_path,
            rate=sample_rate,
            data=concatenated_array
        )
        print(f"Saved final WAV file at {final_wav_path}")

        # Convert final WAV to MP3
        subprocess.run(["ffmpeg", "-i", final_wav_path, mp3_path])
        print(f"Converted WAV to MP3 at {mp3_path}")

        return FileResponse(path=mp3_path, media_type='audio/mpeg', filename=f'{unique_filename}.mp3')

    except Exception as e:
        print("An error occurred:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/")
async def summarize_text(text_to_summarize: TextToSummarize):
    try:
        summary = summarizer(text_to_summarize.text, max_length=130, min_length=30, do_sample=False)
        return {"summary": summary[0]['summary_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_questions/")
async def generate_questions(request: QuestionGenerationRequest):
    try:
        question_generator = QuestionGeneration()
        questions_dict = question_generator.generate_questions_dict(request.document)
        return questions_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
