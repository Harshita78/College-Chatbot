import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))
ignore_punctuations = set(string.punctuation)

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    tokens = nltk.word_tokenize(sentence.lower())
    cleaned_tokens = [
        w for w in tokens
        if w not in ignore_punctuations and w not in stop_words
    ]
    return cleaned_tokens


def lemmatize(word):
    """
    lemmatize = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    -> ["organ", "organ", "organ"]
    """
    return lemmatizer.lemmatize(word)


def bag_of_words(tokenized_sentence, all_words):

    sentence_words = [lemmatize(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag
