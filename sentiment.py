import nltk
import numpy as np 

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bd4 import BeautifulSoup

wordnet_lemmatizer = WordNetLemmatizer()

stopwords = set(w.srtip() for w in open('stopwords.txt'))

positive_reviews = BeautifulSoup(open('electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')

np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]

def my_tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    return tokens

word_index_map = {}
current_index = 0

for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    for token in tokens:
        if token not in word_index_map:
            word_index_map(token) = current_index
            current_index +=1
