import requests
from utils.utils import load_config, create_mcq_quiz

if __name__ == "__main__":
    config = load_config()
    context = """
    Machine organization is the view of the computer that is seen by the logic designer. This includes
    Capabilities & performance characteristics of functional units (e.g., registers, ALU, shifters, etc.). 
    Ways in which these components are interconnected
    How information flows between components
    Logic and means by which such information flow is controlled
    Coordination of functional units to realize the ISA
    Typically the machine organization is designed to meet a given instruction set architecture.
    However, in order to design good instruction sets, it is important to understand how the architecture might be implemented. 

    The functions of the different computer components are
    datapath - performs arithmetic and logic operations 
    e.g., adders, multipliers, shifters
    memory - holds data and instructions
    e.g., cache, main memory, disk
    input - sends data to the computer
    e.g., keyboard, mouse
    output - gets data from the computer
    e.g., screen, sound card
    control - gives directions to the other components
    e.g., bus controller, memory interface unit
    """

    method = "Wordnet"  # or "Sense2Vec"

    url = "http://127.0.0.1:8000/generate_mcq"

    payload = {
        "context": context,
        "method": method
    }

    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        questions_json = response.json()
        mcq_quiz = create_mcq_quiz(questions_json)
        print(mcq_quiz)
    else:
        print(f"Failed to generate MCQ: {response.status_code}, {response.text}")

