import pandas as pd
import itertools
import numpy as np
import string
import re
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
import matplotlib
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB

import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer


##############################################################################


from tensorflow.python.keras.layers import Activation



def get_vectorized_text(vectorizing_method, train_x, test_x, analyzer='word', max_features=None, vectorizing_name = None):
    vectorizer = vectorizing_method(analyzer=analyzer, max_features=max_features)
    train_x_vect = vectorizer.fit_transform(train_x)
    if vectorizing_name is not None:
        with open(f'vectorizers/{vectorizing_name}.pickle', 'wb') as files:
            pickle.dump(vectorizer, files)
    test_x_vect = vectorizer.transform(test_x)
    return train_x_vect, test_x_vect


def predict_labels(vectorizing_method, classification_method, train_x, test_x, train_y, vectorizing_name=None, classification_name=None):
    train_x_vectorized, test_x_vectorized = get_vectorized_text(vectorizing_method, train_x, test_x, vectorizing_name=vectorizing_name)
    classifier = classification_method()
    classifier.fit(train_x_vectorized, train_y)
    if vectorizing_name is not None and classification_name is not None:
        with open(f'models/{vectorizing_name}_{classification_name}.pickle', 'wb') as files:
            pickle.dump(classifier, files)
    predicted_labels = classifier.predict(test_x_vectorized)
    return predicted_labels


def predict_and_get_accuracy(vectorizing_method, classification_method, train_x, test_x, train_y, test_y, vectorizing_name=None, classification_name=None):
    predicted_labels = predict_labels(vectorizing_method, classification_method, train_x, test_x, train_y, vectorizing_name, classification_name)
    return predicted_labels, accuracy_score(test_y, predicted_labels)



def plot_confusion_matrix(cm, classes=[False, True],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


if __name__ == '__main__':
    dataset_df = pd.read_csv('Datasets/preprocessed_dataset.csv')
    print(dataset_df.iloc[0])
    true = dataset_df[dataset_df['label'] == True]

    false = dataset_df[dataset_df['label'] == False]
    print(f'total = {len(dataset_df)}')
    print(f'true = {len(true)}')
    print(f'false = {len(false)}')

    train_x, test_x, train_y, test_y = train_test_split(dataset_df['claim'].values, dataset_df['label'].values, test_size=0.2)

    # no ASCII for arabic - so will convert to unicode
    # train_x = train_x.astype('U')
    # test_x = test_x.astype('U')

    prediction, accuracy = predict_and_get_accuracy(TfidfVectorizer, LogisticRegression, train_x, test_x, train_y, test_y, 'tfidf', 'logisticreg')
    print(f'TFIDF + Logistic Regression Accuracy + : {accuracy}')
    cm = confusion_matrix(test_y, prediction)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cm, title='TFIDF + Logistic Regression, without normalization')
    plt.figure()
    plot_confusion_matrix(cm, normalize=True, title='TFIDF + Logistic Regression, Normalized')
    plt.show()

    prediction, accuracy = predict_and_get_accuracy(CountVectorizer, LogisticRegression, train_x, test_x, train_y, test_y, 'countvec', 'logisticreg')
    print(f'Count Vectorizing + Logistic Regression Accuracy : {accuracy}')
    cm = confusion_matrix(test_y, prediction)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cm, title='Count Vectorizing + Logistic Regression, without normalization')
    plt.figure()
    plot_confusion_matrix(cm, normalize=True, title='Count Vectorizing + Logistic Regression, Normalized')
    plt.show()

    prediction, accuracy = predict_and_get_accuracy(TfidfVectorizer, MultinomialNB, train_x, test_x, train_y, test_y, 'tfidf', 'nb')
    print(f'TFIDF + Naive Bayes Accuracy : {accuracy}')
    cm = confusion_matrix(test_y, prediction)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cm, title='TFIDF + Naive Bayes, without normalization')
    plt.figure()
    plot_confusion_matrix(cm, normalize=True, title='TFIDF + Naive Bayes, Normalized')
    plt.show()

    prediction, accuracy = predict_and_get_accuracy(CountVectorizer, MultinomialNB, train_x, test_x, train_y, test_y, 'countvec', 'nb')
    print(f'Count Vectorizing + Naive Bayes Accuracy : {accuracy}')
    cm = confusion_matrix(test_y, prediction)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cm, title='Count Vectorizing + Naive Bayes, without normalization')
    plt.figure()
    plot_confusion_matrix(cm, normalize=True, title='Count Vectorizing + Naive Bayes, Normalized')
    plt.show()

    list_of_words = []
    for claim in dataset_df.claim:
        for word in claim.split(' '):
            list_of_words.append(word)
    total_words = len(list(set(list_of_words)))

    # creat a tokenizer (task6)
    tokenizer = Tokenizer(num_words=total_words)
    tokenizer.fit_on_texts(train_x)
    train_sequences = tokenizer.texts_to_sequences(train_x)
    test_sequences = tokenizer.texts_to_sequences(test_x)

    padded_train = pad_sequences(train_sequences, maxlen=40, padding='post', truncating='post')
    padded_test = pad_sequences(test_sequences, maxlen=40, truncating='post')

    model = Sequential()

    model.add(Embedding(total_words, output_dim=128))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(1,activation= 'sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())

    y_train = np.asarray(train_y)

    model.fit(padded_train, y_train, validation_split=0.1, epochs=15)

    pred = model.predict(padded_test)

    predection = []
    for i in range(len(pred)):
        if pred[i].item() > 0.5:
            predection.append(1)
        else:
            predection.append(0)

    accuracy = accuracy_score(list(test_y), predection)
    print(f'accuracy = {accuracy}')


    cm = confusion_matrix(predection, test_y)

    # Compute confusion matrix
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, title='LSTM, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, normalize=True, title='LSTM, Normalized')
    plt.show()