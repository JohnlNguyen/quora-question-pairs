import itertools
import json
import re
from collections import Counter

import gensim.models.word2vec
import gensim.models.word2vec
import numpy as np
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

path = lambda x: '~/.kaggle/competitions/quora-question-pairs/' + x
stopwords = set(stopwords.words('english'))
stopwords_exceptions = set(['what', 'when', 'where', 'whether', 'which', 'whose', 'why', 'should', 'how'])
stopwords = stopwords.difference(stopwords_exceptions)


# cleaning data
def my_tokenize(s):
    s = s.lower()  # put into lower case form
    s = re.sub(r"[\'\"\?.()/:,]", ' ', s)  # remove certain punctuations and replace it with space
    s = re.sub(r"\s+", ' ', s)  # remove extra space
    s = re.sub(r"not\s+", 'not_', s)  # deal with negation
    tokens = word_tokenize(s)  # split string into words (tokens)
    porter = PorterStemmer()
    tokens = [porter.stem(t) for t in tokens]  # put words into base form
    tokens = [t for t in tokens if t not in set(stopwords)]  # remove stopwords
    return tokens


def clean_text(df):
    contractions = json.load(open('../data/contractions.json', 'r+'))
    contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))

    def expand_contractions(s, contractions_dict=contractions):
        def replace(match):
            return contractions_dict[match.group(0)]

        return contractions_re.sub(replace, s)

    df['question1'].apply(expand_contractions)
    df['question2'].apply(expand_contractions)
    df['question1'].apply(my_tokenize)
    df['question2'].apply(my_tokenize)
    return df


def gensim_load_vec(path="https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"):
    gensim_emb = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True, limit=500000)
    vocab = gensim_emb.index2word
    shape = gensim_emb.syn0.shape
    return gensim_emb, shape, vocab


def map_word_frequency(document):
    return Counter(itertools.chain(*document))


def sentence2vec(sentence_list, word_counts, embedding_size, word_emb_model, a=1e-3):
    sentence_set = []
    for sentence in sentence_list:
        vs = np.zeros(embedding_size)
        sentence_length = len(sentence)
        for word in sentence:
            a_value = a / (a + word_counts[word])  # smooth inverse frequency, SIF
        try:
            vs = np.add(vs, np.multiply(a_value, word_emb_model[word]))  # vs += sif * word_vector
        except Exception as e:
            pass
        # vs = np.log(vs) - np.log(sentence_length)
        vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)
    return sentence_set


def tokenize_sentences(cleaned_train):
    sent_list = []
    for q1, q2 in zip(cleaned_train['question1'], cleaned_train['question2']):
        token1 = word_tokenize(" ".join(q1))
        token2 = word_tokenize(" ".join(q2))
        sent_list.append(token1)
        sent_list.append(token2)
    return sent_list


def main():
    train = pd.read_csv(path('train.csv'))[:1001]
    test = pd.read_csv(path('test.csv'))[:1001]
    train.dropna(subset=['question1', 'question2'], inplace=True)
    cleaned_train = clean_text(train)
    cleaned_test = clean_text(test)
    model, shape, vocab = gensim_load_vec()
    sent_test = tokenize_sentences(cleaned_test)
    sent_train = tokenize_sentences(cleaned_train)
    word_counts = map_word_frequency(sent_test + sent_train)
    sent_emb = sentence2vec(sent_test, word_counts, 300, model)
    is_dup = []
    for i in range(0, len(sent_test) - 1, 2):
        is_dup.append(float(cosine_similarity([sent_emb[i]], [sent_emb[i + 1]])))
    threshold = .9
    y_pred = [1 if score > threshold else 0 for score in is_dup]
    y_true = train.loc[:1001, 'is_duplicate']
    print(f'accuracy {accuracy_score(y_true, y_pred)} threshold {threshold}')


if __name__ == "__main__":
    main()
