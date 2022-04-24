from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def TF_IDF(train_x, test_x):
    tfidf_vect = TfidfVectorizer(analyzer='word')
    train_x_tfidf = tfidf_vect.fit_transform(train_x)
    test_x_tfidf = tfidf_vect.transform(test_x)


    # word level TF-IDF
    # tfidf_vect = TfidfVectorizer(analyzer='word', max_features=5000)
    # tfidf_vect.fit(train_x.astype('U'))
    # xtrain_tfidf = tfidf_vect.transform(train_x.astype('U')).toarray()
    # xvalid_tfidf = tfidf_vect.transform(test_x.astype('U')).toarray()
    # pickle.dump(tfidf_vect, open("vectorizer.pickle", "wb"))
    # mdl2 = pickle.load(open("vectorizer.pickle", "rb"))
    # pd.DataFrame(xtrain_tfidf[0]).to_csv("1.csv")
    return train_x_tfidf, test_x_tfidf


def count_vectorizer(train_x, test_x):
    count_vect = CountVectorizer(analyzer='word')
    train_x_count_vect = count_vect.fit_transform(train_x)
    test_x_count_vect = count_vect.transform(test_x)


    # count_vec = CountVectorizer(analyzer='word', )
    # count_train = count_vec.fit(train_x.astype('U'))
    # bag_of_words_train = count_vec.transform(train_x.astype('U')).toarray()
    # bag_of_words_test = count_vec.transform(test_x.astype('U')).toarray()
    # pickle.dump(count_vec, open("vectorizer.pickle", "wb"))
    # mdl2 = pickle.load(open("vectorizer.pickle", "rb"))
    # pd.DataFrame(bag_of_words_train[0]).to_csv("2.csv")

    return train_x_count_vect, test_x_count_vect


def get_vectorized_text(vectorizing_method, train_x, test_x, analyzer='word', max_features=None):
    vectorizer = vectorizing_method(analyzer=analyzer, max_features=max_features)
    train_x_vect = vectorizer.fit_transform(train_x)
    test_x_vect = vectorizer.transform(test_x)
    return train_x_vect, test_x_vect




