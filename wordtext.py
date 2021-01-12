'''    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.5)(flat)
    main_output = Dense(4, activation='sigmoid')(drop)'''

import pyodbc
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import math
import datetime
from sklearn import model_selection
import numpy as np
import keras
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import Adam
from keras.losses import MeanSquaredError, BinaryCrossentropy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Conv1D, MaxPooling1D, concatenate, Flatten, Embedding, Dropout
from keras.initializers import Constant
from keras.preprocessing.sequence import pad_sequences


def readserver():
    conn = pyodbc.connect('DRIVER={SQL Server};SERVER=LAPTOP-HUT5SS6N;DATABASE=Independ;Trusted_Connection=yes')
    cursor = conn.cursor()
    cursor.execute(
        "SELECT TOP(2029) [tokenstext],[neg],[neu],[pos],[compound],[lengthmodify],[similarity],[Leftposition],[Rightposition],[Centerposition],[reliability]FROM [Independ].[dbo].[Thedataset201202]  order by reliability ")
    return cursor.fetchall()

def TextCNN_model_1(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test):
    main_input = Input(shape=(500,), dtype='float64')
    embedder = Embedding(len(vocab) + 1, 300, input_length=500, trainable=False)
    embed = embedder(main_input)
    cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=48)(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=47)(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=46)(cnn3)
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    main_output = Dense(2, activation='softmax')(drop)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=2)
    model.fit(x_train_padded_seqs, one_hot_labels, batch_size=800, epochs=10)
    result = model.predict(x_test_padded_seqs)
    result_labels = np.argmax(result, axis=1)
    y_predict = list(map(str, result_labels))
    print('Accuracy', accuracy_score(y_test, y_predict))
    print('f1-score:', f1_score(y_test, y_predict, average='weighted'))

if __name__=='__main__':
    for i in range(10):
        text = []
        label = []
        traintext = []
        trainlabel = []
        data = readserver()
        for i in range(len(data)):
            text.append(data[i].tokenstext)
            label.append(data[i].reliability)
            if i < 2029:
                traintext.append(data[i].tokenstext)
                trainlabel.append(data[i].reliability)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(text)
        vocab = tokenizer.word_index
        x_train, x_test, y_train, y_test = model_selection.train_test_split(traintext, trainlabel, test_size=0.3)

        x_train_word_ids = tokenizer.texts_to_sequences(x_train)
        x_test_word_ids = tokenizer.texts_to_sequences(x_test)

        x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=500)
        x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=500)
        TextCNN_model_1(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test)







