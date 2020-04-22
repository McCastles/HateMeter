#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:25:39 2020

@author: fractum
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 04:54:20 2020

@author: strai
"""


import nltk
from keras.layers import Activation, Dense, Dropout, Embedding, MaxPooling1D, LSTM
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from gensim.models import Word2Vec
import numpy as np
import logging
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import time
import pickle

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
nltk.download('stopwords')

#%%

# WORD2VEC 
W2V_SIZE = 300

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)
decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}

#%% Loading
w2v_model = Word2Vec.load('model.w2v')

#%%


def decode(score):   
    label = NEUTRAL
    if score <= SENTIMENT_THRESHOLDS[0]:
        label = NEGATIVE
    elif score >= SENTIMENT_THRESHOLDS[1]:
        label = POSITIVE
    return label
    
def predict(text, include_neutral=True):
    start_at = time.time()
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    score = model.predict([x_test])[0]
    label = decode(score)

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}  


callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]
#%% Preprocess


vocab_size = 290419
#%%  Tokenize


pickle_in = open("tokenizer.pkl","rb")
tokenizer = pickle.load(pickle_in)

vocab_size = len(tokenizer.word_index) + 1
labels = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']

pickle_in = open("encoder.pkl","rb")
encoder = pickle.load(pickle_in)


#%% Embedding
embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        try:
            embedding_matrix[i] = w2v_model.wv[word]
        except IndexError:
            print('Fuckoff')
            
embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH, trainable=False)

#%% Model

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#%% Loading weights
model.load_weights('model.h5')

#%% Compile model

model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])


#%% test
print(predict("I have a good mood today"))
print(predict("The weather was awful"))
print(predict("Something neutral"))

