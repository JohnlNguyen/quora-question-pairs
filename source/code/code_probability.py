import pandas as pd
import time

import re
import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk import word_tokenize

import json

def prob_cal(str_li):
    """
    :param str_li: list of question to calculate probability of words
    :return: dict - dict with probability of respective words
    """
    BRACKETS_RE = re.compile('[/(){}\[\]\|@,;]:')
    STOPWORDS = stopwords.words('english')
    STOPWORDS.extend([',','?',':','(',')','.','``',"''","'s","n't"])

    def clean_and_tokenize_body(text):
        text = re.sub(BRACKETS_RE, ' ', text.lower())
        return ' '.join([w for w in word_tokenize(text) if not w in STOPWORDS])

    str_li_clean = [clean_and_tokenize_body(i) for i in str_li]
    str_li_biglist = ' '.join(str_li_clean).split()
    word_count = pd.Series(str_li_biglist).value_counts()

    df = pd.DataFrame(word_count)
    df = df.apply(lambda x: x/len(word_count))

    prob_dict = df.to_dict()[0]
    return(prob_dict)

train_data = pd.read_csv('/Users/esmondchu/Dropbox/UC_Davis/STAT/STA141CSpring18/Final_Project/train.csv')

que1 = list(train_data.question1)
del que1[363362]
que2 = list(train_data.question2)
for i in que2:
    if type(i) == float:
        del que2[que2.index(i)]

q1_dict = prob_cal(que1)
q2_dict = prob_cal(que2)

import json
json = json.dumps(q1_dict)
f = open("q1_dict.json","w")
f.write(json)
f.close()

import json
json = json.dumps(q2_dict)
f = open("q2_dict.json","w")
f.write(json)
f.close()

que = que1 + que2

import json
json = json.dumps(que)
f = open("que.json","w")
f.write(json)
f.close()
