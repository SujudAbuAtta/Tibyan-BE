from Datasets.DataPreProcessor import process_text
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def classify_news(news_text):

    processed_text = process_text(news_text)
    print(processed_text)
    logisticRegClassifier, multiNBClassifier, tfidf_vect = build_model()

    x_tfidf = tfidf_vect.transform([processed_text])

    pred1 = logisticRegClassifier.predict(x_tfidf)
    pred2 = multiNBClassifier.predict(x_tfidf)
    print(pred1)
    print(pred2)


def build_model():
    dataset_df = pd.read_csv('Datasets/preprocessed_dataset.csv')
    tfidf_vect = TfidfVectorizer(analyzer='word', max_features=5000)
    tfidf_vect.fit(dataset_df['claim'].values.astype('U'))
    x_tfidf = tfidf_vect.transform(dataset_df['claim'].values.astype('U')).toarray()
    logisticRegClassifier = LogisticRegression().fit(x_tfidf, dataset_df['label'].values)
    multiNBClassifier = MultinomialNB().fit(x_tfidf, dataset_df['label'].values)

    return logisticRegClassifier, multiNBClassifier, tfidf_vect


classify_news("مرحبا ةيةية انا سجود")
