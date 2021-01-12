
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
from keras.losses import MeanSquaredError,BinaryCrossentropy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential,Model
from keras.layers import Input, Dense, Activation, Conv1D,MaxPooling1D, concatenate,Flatten,Embedding,Dropout
from keras.initializers import Constant
def readserver():
    conn = pyodbc.connect('DRIVER={SQL Server};SERVER=LAPTOP-HUT5SS6N;DATABASE=Independ;Trusted_Connection=yes')
    cursor = conn.cursor()
    cursor.execute("SELECT TOP(2029) [tokenstext],[neg],[neu],[pos],[compound],[lengthmodify],[similarity],[Leftposition],[Rightposition],[Centerposition],[reliability]FROM [Independ].[dbo].[Thedataset201202]  order by reliability ")
    return cursor.fetchall()



def makemartix(data):
    tfidftext=[]
    featuresize=500

    labelnumeric=[]
    unlabelnumeric=[]

    labeltfidf=[]
    unlabeltfidf=[]


    labelX=[]
    unlabelX=[]



    for i in range(len(data)):
        tfidftext.append(data[i].tokenstext)
    Tfidf_vect = TfidfVectorizer(max_features=featuresize)
    tfidf=Tfidf_vect.fit_transform(tfidftext)
    matrix=tfidf.toarray()
    j=0
    Traintext=[]

    Trainnumeric=[]

    Testtext=[]
    Testnumeric=[]



    label=[]
    for i in range(len(data)):
        if i<2029:
            labelnumeric.append([float(data[i].neg),float(data[i].neu),float(data[i].pos),float(data[i].compound),float(data[i].lengthmodify),float(data[i].similarity),float(data[i].Leftposition),float(data[i].Rightposition),float(data[i].Centerposition)])
            labeltfidf.append(matrix[i])
            label.append(data[i].reliability)
            labelX.append([labelnumeric[i],labeltfidf[i]])

        else:
            unlabelnumeric.append([float(data[i].neg),float(data[i].neu),float(data[i].pos),float(data[i].compound),float(data[i].lengthmodify),float(data[i].similarity),float(data[i].Leftposition),float(data[i].Rightposition),float(data[i].Centerposition)])
            unlabeltfidf.append(matrix[i])
            unlabelX.append([unlabelnumeric[j],unlabeltfidf[j]])
            j=j+1


    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(labeltfidf, label, test_size=0.2)





    return    Train_X, Test_X, Train_Y, Test_Y






def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float64)
  return arg
def my_funn(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float64)
  return arg


def CNN(Train_X, Test_X, Train_Y, Test_Y):
    tf.keras.backend.clear_session()
    #shareCNN
    featuresize=500

    main_input = Input(shape=(featuresize,), dtype='float64')


    embedder = Embedding(featuresize+ 1, 300, input_length=featuresize, trainable=False)
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

    model=Model(inputs=main_input,outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    one_hot_labels = keras.utils.to_categorical(Train_Y, num_classes=2)



    value_1 = my_func(tf.constant(Train_X))

    model.fit(value_1, one_hot_labels, batch_size=800, epochs=10)
    value_2 = my_func(tf.constant(Test_X))
    result = model.predict(value_2)
    result_labels = np.argmax(result, axis=1)
    y_predict = list(map(str, result_labels))
    print('Accuracy', accuracy_score(Test_Y, y_predict))
    print('f1-score:', f1_score(Test_Y, y_predict, average='weighted'))













''' 
   
    # Assign it as tfe.variable since we will change it across epochs
    learning_rate = tfe.Variable(max_learning_rate)
    beta_1 = tfe.Variable(initial_beta1)

    # Download and Save Dataset in Tfrecords
    loader = SvnhLoader('./data', NUM_TRAIN_SAMPLES,
                        num_validation_samples, num_labeled_samples)
    loader.download_images_and_generate_tf_record()

    # Generate data loaders
    train_labeled_iterator, train_unlabeled_iterator, validation_iterator, test_iterator = loader.load_dataset(
        batch_size, epochs)
   
   
   # Evaluate on the final test set
    num_test_batches = math.ceil(NUM_TEST_SAMPLES / batch_size)
    test_accuracy = tfe.metrics.Accuracy()
    for test_batch in range(num_test_batches):
        X_test, y_test, _ = test_iterator.get_next()
        y_test_predictions = model(X_test, training=False)
        test_accuracy(tf.argmax(y_test_predictions, 1), tf.argmax(y_test, 1))

    print("Final Test Accuracy: {:.6%}".format(test_accuracy.result()))'''


'''    model = Model(inputs=main_input, outputs=main_output)   
 model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
'''


if __name__ == '__main__':
    co = readserver()

    for i in range(10):
        a, b, c, d = makemartix(co)
        CNN(a, b, c, d)

