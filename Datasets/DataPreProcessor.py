# First Step: Pre-Processing Text.
# 1.Removing Punctuations
# 2.Removing Diacritics
# 3.Change same letters in arabic to one letter
# 4.Removing stopwords
# 5.Snowball Stemming

import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


arabic_stop_words = set(stopwords.words('arabic'))
arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|"!”…“–ـ'''
other_punctuations = string.punctuation
all_punctuations = arabic_punctuations + other_punctuations


def remove_punctuations(text):
    new_text = str.maketrans(' ', ' ', all_punctuations)
    return text.translate(new_text)


def unify_letters(text):
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    return text


def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text


def remove_stopwords(text):
    text_tokens = word_tokenize(text)
    text = [word for word in text_tokens if word not in arabic_stop_words]
    return text


def process_text(text):
    text = remove_punctuations(text).rstrip()
    text = remove_stopwords(text)
    processed_text = list()
    for word in text:
        processed_word = remove_diacritics(word)
        processed_word = SnowballStemmer("arabic").stem(processed_word)
        processed_word = unify_letters(processed_word)
        processed_text.append(processed_word)
    return ' '.join(processed_text)


def process_dataset(df):
    for index, _ in df.iterrows():
        text = df.loc[index, 'claim']
        if type(text) == str:
            df.loc[index, 'claim'] = process_text(text)
    return df[['claim', 'label']]