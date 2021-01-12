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


def readserver():
    conn = pyodbc.connect('DRIVER={SQL Server};SERVER=LAPTOP-HUT5SS6N;DATABASE=Independ;Trusted_Connection=yes')
    cursor = conn.cursor()
    cursor.execute(
        "SELECT TOP(15000) [tokenstext],[neg],[neu],[pos],[compound],[lengthmodify],[similarity],[Leftposition],[Rightposition],[Centerposition],[reliability]FROM [Independ].[dbo].[Thedataset201202]  order by reliability ")
    return cursor.fetchall()


'''    for j in range(len(Train_X)):
        Traintext=Train_X[j][1]
        Trainnumeric=Train_X[j][0]

    for z in range(len(Test_X)):
        Testtext = Test_X[z][1]
        Testnumeric = Test_X[z][0]'''


def makemartix(data):
    tfidftext = []
    featuresize = 527

    labelnumeric = []
    unlabelnumeric = []

    labeltfidf = []
    unlabeltfidf = []

    labelX = []
    unlabelX = []

    for i in range(len(data)):
        tfidftext.append(data[i].tokenstext)
    Tfidf_vect = TfidfVectorizer(max_features=featuresize)
    tfidf = Tfidf_vect.fit_transform(tfidftext)
    matrix = tfidf.toarray()
    j = 0
    Traintext = []

    Trainnumeric = []

    Testtext = []
    Testnumeric = []

    label = []
    for i in range(len(data)):
        if i < 2029:
            labelnumeric.append([  float(data[i].neg), float(data[i].neu), float(data[i].pos), float(data[i].compound),
                                   float(data[i].lengthmodify), float(data[i].similarity), float(data[i].Leftposition),
                                   float(data[i].Rightposition), float(data[i].Centerposition)])
            labeltfidf.append(matrix[i].tolist())
            label.append(data[i].reliability)
            arr = np.array(labelnumeric[i] + labeltfidf[i])
            labelX.append(arr)

        else:
            unlabelnumeric.append([float(data[i].neg), float(data[i].neu), float(data[i].pos), float(data[i].compound),
                                   float(data[i].lengthmodify), float(data[i].similarity), float(data[i].Leftposition),
                                   float(data[i].Rightposition), float(data[i].Centerposition)])
            unlabeltfidf.append(matrix[i])
            unlabelX.append([unlabelnumeric[j], unlabeltfidf[j]])
            j = j + 1








    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(labelX, label, test_size=0.3)




    return Train_X, Test_X, Train_Y, Test_Y



def my_func(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float64)
    return arg


def my_funn(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float64)
    return arg


def CNN(Train_X, Test_X, Train_Y, Test_Y):
    tf.keras.backend.clear_session()
    # shareCNN
    featuresize = 530

    main_input = Input(shape=(featuresize,), dtype='float64')

    embedder = Embedding(featuresize + 1, 300, input_length=featuresize, trainable=False)
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
    output = Dense(2, activation='softmax')(drop)

    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    one_hot_labels = keras.utils.to_categorical(Train_Y, num_classes=2)

    value_1 = my_func(tf.constant(Train_X))

    model.fit(value_1, one_hot_labels, batch_size=800, epochs=10)
    value_3 = my_func(tf.constant(Test_X))

    result = model.predict(value_3)

    result_labels = np.argmax(result, axis=1)
    y_predict = list(map(str, result_labels))
    print('Accuracy', accuracy_score(Test_Y, y_predict))
    print('f1-score:', f1_score(Test_Y, y_predict, average='weighted'))





if __name__ == '__main__':
    co = readserver()
    for i in range(10):
        Train_X, Test_X, Train_Y, Test_Y = makemartix(co)
        CNN(Train_X, Test_X, Train_Y, Test_Y)

