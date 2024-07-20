import random
import re
import nltk
import string
import yaml
import numpy as np
from flashtext import KeywordProcessor
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity
from similarity.normalized_levenshtein import NormalizedLevenshtein
import pke
from pathlib import Path

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

parent_dir = Path(__file__).parent.parent 
config_file = parent_dir / "config" / "models_config.yaml"
def load_config(path=config_file):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def postprocesstext(content):
    final = ""
    for sent in nltk.sent_tokenize(content):
        sent = sent.capitalize()
        final = final + " " + sent
    return final.strip()

def get_keywords(originaltext, summarytext):
    keywords = get_nouns_multipartite(originaltext)
    keyword_processor = KeywordProcessor()
    for keyword in keywords:
        keyword_processor.add_keyword(keyword)

    keywords_found = keyword_processor.extract_keywords(summarytext)
    keywords_found = list(set(keywords_found))

    important_keywords = []
    for keyword in keywords:
        if keyword in keywords_found:
            important_keywords.append(keyword)

    return important_keywords[:4]

def get_nouns_multipartite(content):
    out = []
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content, language='en')
        pos = {'PROPN', 'NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += nltk.corpus.stopwords.words('english')
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
        keyphrases = extractor.get_n_best(n=15)

        for val in keyphrases:
            out.append(val[0])
    except Exception as ex:
        out = []

    return out

def filter_same_sense_words(original, wordlist):
    filtered_words = []
    base_sense = original.split('|')[1]
    for eachword in wordlist:
        if eachword[0].split('|')[1] == base_sense:
            filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
    return filtered_words

def get_highest_similarity_score(wordlist, wrd):
    normalized_levenshtein = NormalizedLevenshtein()
    score = []
    for each in wordlist:
        score.append(normalized_levenshtein.similarity(each.lower(), wrd.lower()))
    return max(score)

def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        mmr = (lambda_param) * candidate_similarities - (1-lambda_param) * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

def get_distractors_wordnet(word):
    distractors = []
    try:
        syn = wn.synsets(word, 'n')[0]

        word = word.lower()
        orig_word = word
        if len(word.split()) > 0:
            word = word.replace(" ", "_")
        hypernym = syn.hypernyms()
        if len(hypernym) == 0:
            return distractors
        for item in hypernym[0].hyponyms():
            name = item.lemmas()[0].name()
            if name == orig_word:
                continue
            name = name.replace("_", " ")
            name = " ".join(w.capitalize() for w in name.split())
            if name is not None and name not in distractors:
                distractors.append(name)
    except:
        print("Wordnet distractors not found")
    return distractors


def create_mcq_quiz(questions_json):
    mcq_quiz = []

    for idx, question_obj in enumerate(questions_json):
        question = question_obj['question']
        correct_answer = question_obj['correct_answer']
        distractors = question_obj['distractors']

        # Shuffle correct answer and distractors to randomize options
        options = [correct_answer] + distractors
        random.shuffle(options)

        # Create MCQ question format
        mcq_question = f"Q{idx + 1}: {question}\n"
        options_letters = ['A', 'B', 'C', 'D'][:len(options)]  # Limit to 4 options
        for i, (option_letter, option) in enumerate(zip(options_letters, options)):
            mcq_question += f"{option_letter}. {option}"
            if i < len(options) - 1:
                mcq_question += "\n"

        mcq_quiz.append(mcq_question)

    return "\n\n".join(mcq_quiz)
    


def slice_array_wave(input_array, amplitude_threshold, time_threshold, ignore_initial_steps=0):
    """
    Slice an input array based on consecutive low amplitude values.

    Parameters:
    - input_array (numpy.ndarray): Input array containing amplitude values.
    - amplitude_threshold (float): Threshold below which amplitudes are considered low.
    - time_threshold (int): Number of consecutive low amplitude values needed to trigger slicing.
    - ignore_initial_steps (int, optional): Number of initial steps to ignore before checking for consecutive low amplitudes.

    Returns:
    numpy.ndarray: Sliced array up to the point where consecutive low amplitudes reach the specified time_threshold.
    """

    low_amplitude_indices = np.abs(input_array) < amplitude_threshold

    consecutive_count = 0
    for i, is_low_amplitude in enumerate(low_amplitude_indices[ignore_initial_steps:]):
        if is_low_amplitude:
            consecutive_count += 1
        else:
            consecutive_count = 0

        if consecutive_count >= time_threshold:
            return input_array[:i + int(time_threshold/4)]

    return input_array


def split_sentences(text):
    sentences = re.split(r'\. |\.\n|\.\n\n|!|\?|;', text)
    sentences = [sentence.strip() + '..' for sentence in sentences]
    sentences = list(filter(None, sentences))
    number_of_sentences = len(sentences)
    return sentences[:-1], number_of_sentences - 1