from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from TextToFeatures import get_vectorized_text

def logistic_regression_prediction(text_to_features_method, train_x, test_x, train_y):
    train_x_vectorized, test_x_vectorzed = text_to_features_method(train_x, test_x)
    classifier = LogisticRegression()
    classifier.fit(train_x_vectorized, train_y)

    # save resulting model

    # pickle.dump(classifier, open("logistic_reg.pickle", "wb"))

    # Predicting the test set results

    predicted_labels = classifier.predict(test_x_vectorzed)
    return predicted_labels


def naive_bayes_prediction(text_to_features_method, train_x, test_x, train_y):
    train_x_vectorized, test_x_vectorzed = text_to_features_method(train_x, test_x)
    classifier = MultinomialNB()


def predict_labels(vectorizing_method, classification_method, train_x, test_x, train_y):
    train_x_vectorized, test_x_vectorized = get_vectorized_text(vectorizing_method, train_x, test_x)
    classifier = classification_method()
    classifier.fit(train_x_vectorized, train_y)
    predicted_labels = classifier.predict(test_x_vectorized)
    return predicted_labels
