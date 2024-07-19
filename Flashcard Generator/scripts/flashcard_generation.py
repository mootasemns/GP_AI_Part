import re
import yaml
from nltk import sent_tokenize, word_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

nltk.download('punkt')

class FlashcardGeneration:
    '''This class contains the method to generate flashcards'''

    def __init__(self):
        current_path = Path("/content/config.yaml").parent
        config_file_path = current_path  / "config.yaml"
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)

        self.num_flashcards = config['flashcard_generation']['num_flashcards']
        fine_tuned_model_directory = config['flashcard_generation']['fine_tuned_model_directory']

        self.model = T5ForConditionalGeneration.from_pretrained(fine_tuned_model_directory)
        self.tokenizer = T5Tokenizer.from_pretrained(fine_tuned_model_directory, legacy=False)

    def clean_text(self, text):
        # Remove newline characters and split the text into sentences
        text = text.replace('\n', ' ')  # Remove newline chars
        sentences = sent_tokenize(text)
        cleaned_text = ""
        for sentence in sentences:
            # Remove non-alphanumeric chars
            cleaned_sentence = re.sub(r'([^\s\w]|_)+', '', sentence)

            # Substitute multiple spaces with a single space
            cleaned_sentence = re.sub(' +', ' ', cleaned_sentence)
            cleaned_text += cleaned_sentence

            # Ensure sentences end with a period
            if cleaned_text[-1] == ' ':
                cleaned_text = cleaned_text[:-1] + '.'
            else:
                cleaned_text += '.'

            cleaned_text += ' '  # Pad with a space at the end
        return cleaned_text

    def extract_key_terms(self, document):
        '''Extract key terms using TF-IDF vectorization'''
        sentences = sent_tokenize(document)
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        sorted_terms = sorted(zip(tfidf_matrix.sum(axis=0).tolist()[0], feature_names), reverse=True)
        key_terms = [term for score, term in sorted_terms[:self.num_flashcards]]
        return key_terms

    def generate_flashcards_dict(self, document):
        document = self.clean_text(document)
        key_terms = self.extract_key_terms(document)

        flashcards_dict = {}
        for i, term in enumerate(key_terms[:self.num_flashcards]):
            definition = self.extract_definition(term, document)
            flashcards_dict[i + 1] = {
                "term": term,
                "definition": definition
            }

        return flashcards_dict

    def extract_definition(self, term, document):
        # Simple heuristic for definition extraction based on the term
        sentences = sent_tokenize(document)
        for sentence in sentences:
            if term.lower() in sentence.lower():
                return sentence.strip()  # Return the whole sentence as definition

        return "Definition not found"  # Fallback if definition extraction fails

# Example usage
if __name__ == "__main__":
    document = """
    The solar system consists of the Sun and the objects that orbit it, including eight planets, moons, asteroids, comets, and meteoroids. The Sun is the star at the center of the solar system, providing light and heat to the planets. The four inner planets—Mercury, Venus, Earth, and Mars—are terrestrial planets, meaning they have solid rocky surfaces. The four outer planets—Jupiter, Saturn, Uranus, and Neptune—are gas giants, composed mainly of hydrogen and helium.

    Earth's atmosphere is composed primarily of nitrogen and oxygen and supports a diverse range of life forms. The planet's surface is approximately 71% water, which is essential for life. Earth's orbit around the Sun, combined with its axial tilt, results in the changing seasons.

    The Moon, Earth's only natural satellite, influences the planet's tides and has phases that are observable from Earth. The lunar surface is covered with craters and maria, which are large, dark, basaltic plains formed by ancient volcanic eruptions.

    Jupiter is the largest planet in the solar system and has a strong magnetic field. It is known for its Great Red Spot, a giant storm that has been observed for over 300 years. Saturn is famous for its extensive ring system, which is made up of ice and rock particles.

    Space exploration has expanded our understanding of the solar system. The Voyager missions, launched by NASA, have provided valuable data about the outer planets and their moons. Advances in technology continue to enhance our ability to explore and study space, revealing new insights about our place in the universe.
    """

    flashcard_generator = FlashcardGeneration()
    flashcards_dict = flashcard_generator.generate_flashcards_dict(document)

    for num, info in flashcards_dict.items():
        print(f"Term {num}: {info['term']}")
        print(f"Definition: {info['definition']}")

