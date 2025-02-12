from pathlib import Path
import re
import nltk
import yaml
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load configuration from config.yaml
ccurrent_path = Path(__file__).resolve().parent.parent
config_file_path = ccurrent_path / "config.yaml"

with open(config_file_path, 'r') as f:
    config = yaml.safe_load(f)

# Download the necessary nltk resources
nltk.download('stopwords')
nltk.download('punkt')

class TermDefinitionExtractor:
    ''' This class contains all the methods required for extracting terms and definitions from a given document '''

    def __init__(self):
        self.num_terms = config['term_extraction']['num_terms']
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        self.model = T5ForConditionalGeneration.from_pretrained(config['term_extraction']['model_checkpoint'])
        self.tokenizer = T5Tokenizer.from_pretrained(config['term_extraction']['model_checkpoint'])
        self.term_definition_dict = dict()

    def get_term_definition_dict(self, document):
        '''
        Returns a dict of terms and definitions in the format:
        term_number: {
            term: str,
            definition: str
        }
        Params:
            * document: string
        Returns:
            * dict
        '''
        # Find candidate keywords
        self.candidate_keywords = self.get_candidate_keywords(document)

        # Set tfidf scores before ranking candidate keywords
        self.set_tfidf_scores(document)

        # Rank the keywords using calculated tf-idf scores
        self.rank_keywords()

        # Form the terms and definitions based on ranked keywords
        self.form_terms_and_definitions(document)

        return self.term_definition_dict

    def get_filtered_sentences(self, document):
        ''' Returns a list of sentences - each of which has been cleaned of stopwords.
        Params:
                * document: a paragraph of sentences
        Returns:
                * list<str> : list of string
        '''
        sentences = sent_tokenize(document)  # Split the document into sentences
        return [self.filter_sentence(sentence) for sentence in sentences] # Filter stopwords from each sentence and return

    def filter_sentence(self, sentence):
        '''Returns the sentence without stopwords
        Params:
                * sentence: A string
        Returns:
                * string
        '''
        words = word_tokenize(sentence) # Split the sentence into words
        return ' '.join(w for w in words if w not in self.stop_words) # Join words that are not stopwords into a sentence and return

    def get_candidate_keywords(self, document):
        ''' Returns a list of keywords based on tf-idf scores
        Params:
                * document : string
        Returns:
                * list<str>
        '''
        filtered_sentences = self.get_filtered_sentences(document)
        tf_idf_vector = self.vectorizer.fit_transform(filtered_sentences)
        feature_names = self.vectorizer.get_feature_names_out()
        tf_idf_matrix = tf_idf_vector.todense().tolist()

        keyword_scores = dict()

        # Calculate the score for each keyword
        for i in range(len(feature_names)):
            keyword = feature_names[i]
            score = sum(row[i] for row in tf_idf_matrix)
            keyword_scores[keyword] = score

        sorted_keywords = sorted(keyword_scores, key=keyword_scores.get, reverse=True)
        return sorted_keywords[:self.num_terms]

    def set_tfidf_scores(self, document):
        ''' Sets the tf-idf scores for each word'''
        self.unfiltered_sentences = sent_tokenize(document)
        self.filtered_sentences = self.get_filtered_sentences(document)

        self.word_score = dict()
        self.sentence_for_max_word_score = dict()

        tf_idf_vector = self.vectorizer.fit_transform(self.filtered_sentences)
        feature_names = self.vectorizer.get_feature_names_out()
        tf_idf_matrix = tf_idf_vector.todense().tolist()

        num_sentences = len(self.unfiltered_sentences)
        num_features = len(feature_names)

        # Calculate avg score for each word
        for i in range(num_features):
            word = feature_names[i]
            self.sentence_for_max_word_score[word] = ""
            tot = 0.0
            cur_max = 0.0

            for j in range(num_sentences):
                tot += tf_idf_matrix[j][i]

                if tf_idf_matrix[j][i] > cur_max:
                    cur_max = tf_idf_matrix[j][i]
                    self.sentence_for_max_word_score[word] = self.unfiltered_sentences[j]

            self.word_score[word] = tot / num_sentences

    def rank_keywords(self):
        '''Rank keywords according to their score'''
        self.candidate_triples = []

        for candidate_keyword in self.candidate_keywords:
            self.candidate_triples.append([
                self.get_keyword_score(candidate_keyword),
                candidate_keyword,
                self.get_corresponding_sentence_for_keyword(candidate_keyword)
            ])

        self.candidate_triples.sort(reverse=True)

    def get_keyword_score(self, keyword):
        ''' Returns the score for a keyword
        Params:
            * keyword : string of possible several words
        Returns:
            * float : score
        '''
        score = 0.0

        # Calculate total score for the keyword
        for word in word_tokenize(keyword):
            if word in self.word_score:
                score += self.word_score[word]
        return score

    def get_corresponding_sentence_for_keyword(self, keyword):
        ''' Finds and returns a sentence containing the keywords'''
        words = word_tokenize(keyword)
        for word in words:
            if word not in self.sentence_for_max_word_score:
                continue

            sentence = self.sentence_for_max_word_score[word]

            all_present = True
            for w in words:
                if w not in sentence:
                    all_present = False

            if all_present:
                return sentence
        return ""

    def form_terms_and_definitions(self, document):
        ''' Forms the term and definition and populates the term definition dict '''
        used_sentences = list()
        idx = 0
        cntr = 1
        num_candidates = len(self.candidate_triples)
        while cntr <= self.num_terms and idx < num_candidates:
            candidate_triple = self.candidate_triples[idx]

            if candidate_triple[2] not in used_sentences:
                used_sentences.append(candidate_triple[2])

                term_definition = self.generate_definition(candidate_triple[1], candidate_triple[2])
                self.term_definition_dict[cntr] = {
                    "term": candidate_triple[1],
                    "definition": term_definition
                }

                cntr += 1
            idx += 1

    def generate_definition(self, term, context):
        ''' Generate a definition using T5 model '''
        input_text = f"term: {term}  context: {context} </s>"
        features = self.tokenizer([input_text], return_tensors='pt')

        output = self.model.generate(input_ids=features['input_ids'],
                                     attention_mask=features['attention_mask'],
                                     max_length=config['term_extraction']['max_length'],
                                     num_beams=config['term_extraction']['num_beams'],
                                     early_stopping=True)

        definition = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return definition

# Example usage
document = """
The mitochondrion is known as the powerhouse of the cell. It is responsible for producing the energy currency of the cell, ATP.
Photosynthesis occurs in the chloroplasts of plant cells. It converts light energy into chemical energy stored in glucose.
"""

extractor = TermDefinitionExtractor()
term_definition_dict = extractor.get_term_definition_dict(document)
print(term_definition_dict)
