import re
from nltk import sent_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer
from extract import QuestionExtractor

class QuestionGeneration:
    '''This class contains the method
    to generate questions
    '''

    def __init__(self, num_questions, num_options):
        self.num_questions = num_questions
        self.num_options = num_options
        fine_tuned_model_directory = "./fine_tuned_model"
        self.model = T5ForConditionalGeneration.from_pretrained(fine_tuned_model_directory)
        self.tokenizer = T5Tokenizer.from_pretrained(fine_tuned_model_directory)
        self.extractor = QuestionExtractor(num_questions)

    def clean_text(self, text):
        # remove newline characters and split the text into sentences
        text = text.replace('\n', ' ')  # remove newline chars
        sentences = sent_tokenize(text)
        cleaned_text = ""
        for sentence in sentences:
            # remove non alphanumeric chars
            cleaned_sentence = re.sub(r'([^\s\w]|_)+', '', sentence)

            # substitute multiple spaces with single space
            cleaned_sentence = re.sub(' +', ' ', cleaned_sentence)
            cleaned_text += cleaned_sentence

            # ensure sentences end with a period
            if cleaned_text[-1] == ' ':
                cleaned_text[-1] = '.'
            else:
                cleaned_text += '.'

            cleaned_text += ' '  # pad with a space at the end
        return cleaned_text

    def generate_questions_dict(self, document):
        document = self.clean_text(document)
        input_ids = self.tokenizer(document, return_tensors="pt", max_length=512, truncation=True)['input_ids']
        
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
        # Simple heuristic for answer extraction based on the generated question
        sentences = sent_tokenize(document)
        for sentence in sentences:
            if question.lower() in sentence.lower():
                return sentence.strip()  # Return the whole sentence as answer
        
        return "Answer not found"  # Fallback if answer extraction fails

# Example usage
# if __name__ == "__main__":
#     document = "Apple is looking at buying U.K. startup for $1 billion. San Francisco considers banning sidewalk delivery robots."
#     question_generator = QuestionGeneration(num_questions=2, num_options=3)
#     questions_dict = question_generator.generate_questions_dict(document)
    
#     for q_num, q_info in questions_dict.items():
#         print(f"Q{q_num}: {q_info['question']}")
#         print(f"Answer: {q_info['answer']}")

