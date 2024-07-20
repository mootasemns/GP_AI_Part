# %%time

# from IPython.display import clear_output

# ! pip install -U accelerate -qq # for inference time optimization (enable_cpu_offload)

# clear_output()

import time
import gc
import re

import numpy as np
import pandas as pd
from scipy.io.wavfile import write # to save audios

### HF
from transformers import AutoProcessor, AutoModel

### torch
import torch
# from torchaudio.transforms import Resample
# import torchaudio

### audio
import transformers
print('transformers version: ', transformers.__version__)
print('torch version: ', torch.__version__)

class CFG:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    ### processor
    SPEAKER = "v2/en_speaker_0" # voice preset

    ### model
    MODEL_NAME = 'suno/bark'

    ### post-processing: to visualize and remove Noise
    AMPLITUDE_THRESHOLD = 0.05
    TIME_THRESHOLD = int(24_000 * 0.5) # sample_rate * n_seconds
    IGNORE_INITIAL_STEPS = int(24_000 * 0.5) # sample_rate * n_seconds


### choose text to make inference on
text_to_infer = '''
Serena is very beautifull. 

'''

len(text_to_infer)



processor = AutoProcessor.from_pretrained(
    CFG.MODEL_NAME,
    voice_preset = CFG.SPEAKER,
    return_tensors = 'pt'
)
model = AutoModel.from_pretrained(
    CFG.MODEL_NAME,
    torch_dtype = torch.float16, # half-precision
).to(CFG.DEVICE);

### inference optimization with accelerate
model.enable_cpu_offload()

# clear_output()

### bark architecture
model.eval()


from torch.nn import Parameter

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) + \
           sum(p.numel() for p in model.parameters() if not p.requires_grad and isinstance(p, Parameter))

n_params = count_parameters(model)
### suno/bark: 1 B params
### suno/bark-small: 410 M params

def count_tokens(text, processor = processor):
    return len(processor.tokenizer(text)['input_ids'])


total_tokens = count_tokens(text_to_infer, processor = processor)

def split_sentences(text):
    '''
    Given a text, return a list of sentences.
    '''
    ### Split on '. ', '.\n', '.\n\n', '!', '?', and ';'
    sentences = re.split(r'\. |\.\n|\.\n\n|!|\?|;', text)

    ### Strip whitespace from each sentence
    sentences = [
        sentence.strip() + '..'
        for sentence in sentences
    ]

    ### Remove empty strings from the list of sentences
    sentences = list(filter(None, sentences))

    number_of_sentences = len(sentences)

    return sentences[:-1], number_of_sentences - 1


### testing function
sentences, number_of_sentences = split_sentences(text = text_to_infer)
print(f'Sentences: \n {sentences}')
print(f'\n\nNumber of sentences: {number_of_sentences}')



### prepare list of sentences
sentences, number_of_sentences = split_sentences(text = text_to_infer)
print(f'\nSentences in this text:\n {sentences}')
print(f'\nNumber of sentences in this text: {number_of_sentences}\n')

all_audio_arrays = []
all_times = []

### inference per sentence
for sentence_number in range(number_of_sentences):

    current_sentence = sentences[sentence_number]

    print(f'Processing sentence {sentence_number + 1}/{number_of_sentences}...')

    start_time = time.time()

    ### prepare input for the model, call the processor for the current sentence only
    inputs = processor(
        text = current_sentence,
        return_tensors = "pt",
        return_attention_mask = True,
        max_length = 1024,
        voice_preset = CFG.SPEAKER,
        add_special_tokens = False,
    ).to(CFG.DEVICE)

    ### count tokens
    n_tokens = count_tokens(current_sentence)

    ### model inference
    with torch.inference_mode():
        result = model.generate(
            **inputs,
            do_sample = True,
            semantic_max_new_tokens = 1024,
            pad_token_id = processor.tokenizer.pad_token_id,
        )

    ### save results
    all_audio_arrays.append(result.cpu().numpy())
    elapsed_time = round((time.time() - start_time), 2)
    all_times.append(elapsed_time)

    sentences_left_to_processs = number_of_sentences - (sentence_number + 1)
    average_time = np.array(all_times).mean()
    time_to_complete = round((sentences_left_to_processs * average_time)/60, 2) # in minutes

    print(f'''
          Sentence {sentence_number + 1}/{number_of_sentences} processed:
          \tNumber of tokens in sentence: {n_tokens}
          \tLength of sentence: {len(current_sentence)}
          \tNumber of sentences in text: {number_of_sentences}
          \tShape of tensor for this sentence: {result.size()}
          \tElapsed time for this sentence: {elapsed_time} s
          \tEstimated time to complete: {time_to_complete} min
          ''')

    sample_rate = model.generation_config.sample_rate # 24_000

    result_array = result.cpu().numpy()
    result_len = result_array.shape[-1]

avg_time = np.array(all_times).mean().round(2) # average time per sentence (in seconds)
avg_time



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
#             return input_array[:i - time_threshold]
#             return input_array[:i - int(time_threshold/2) + int(time_threshold/4)]
            return input_array[:i + int(time_threshold/4)]

    return input_array


### Concatenate results for each batch
concatenated_array  = np.array([])

for audio_number, sentence_audio in enumerate(all_audio_arrays):

    print(f'Audio {audio_number + 1}/{len(all_audio_arrays)}')

    ### concat audio arrays
    current_array = all_audio_arrays[audio_number].squeeze()

    ### post-processed array (remove padding in inference was done in batches)
    current_array = slice_array_wave(
        input_array = current_array,
        amplitude_threshold = CFG.AMPLITUDE_THRESHOLD,
        time_threshold = CFG.TIME_THRESHOLD,
        ignore_initial_steps = CFG.IGNORE_INITIAL_STEPS
    )

    concatenated_array = np.concatenate([concatenated_array, current_array])

### save as np.array
# np.save(f'saved_audio/final_audio_sr_{sample_rate}.npy', concatenated_array)

### save as .wav file
write(
    f'saved_audio/final_audio_sr_{sample_rate}.wav',
    rate = sample_rate,
    data = concatenated_array
  )

