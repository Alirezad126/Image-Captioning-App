
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from model.vgg import *
#index to word from tokenizer:

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def correction(sequence):
    new_sequence = ""
    explode = sequence.split()
    for i in range(1,len(explode)-1):
        new_sequence += explode[i] + " "
    return new_sequence

# generate caption for an image:
def predict_caption(model, uploaded_file, tokenizer, max_length):
    
    feature = feature_extraction(uploaded_file)

    
    in_text = "startseq"
    #iterate over max length of sequence:
    for i in range(max_length):
        #encode input sequence:
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        #pad the sequence:
        sequence = pad_sequences([sequence], max_length)
        #predict
        yhat = model.predict([feature, sequence], verbose=0)
        #get index with high prob :
        yhat = np.argmax(yhat)
        #convert index to word
        word = idx_to_word(yhat, tokenizer)
        #stop if word not found
        if word is None:
            break

        #append word as input for generating next word
        in_text += " " + word
        #stop if we reach end:
        if word == "endseq":
            break

    return correction(in_text)