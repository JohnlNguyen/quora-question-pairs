
# coding: utf-8

# # Cleaning data

# In[32]:


import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer


# In[34]:


df = pd.read_csv('train.csv')


# In[35]:


# find questions that are empty
df["question1"][363362] = "wrong"
df["question2"][105780] = "wrong"
df["question2"][201841] = "wrong"


# In[36]:


# generate stopword
alist = [line.rstrip() for line in open('stopwords.txt')]
stopword = stopwords.words('english') + alist
stopword = [e for e in stopword if e not in ('what', 'when', 'where','whether','which','whose', 'why','should','how')]


# In[37]:


contractions = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}


# In[38]:


# spread out contraction for question1 and question2
question1 = []
for i in range(len(df)):
    s1 = df['question1'][i]
    tokens = re.findall(r'\S+', s1)
    mylist= []
    for t in tokens:
        if t in contractions:
            t = contractions[t]
        mylist.append(t)
        result1 = " ".join(mylist)
    question1.append(result1)
    i = i+1


question2 = []
for i in range(len(df)):
    s2 = df['question2'][i]
    tokens = re.findall(r'\S+', s2)
    mylist= []
    for t in tokens:
        if t in contractions:
            t = contractions[t]
        mylist.append(t)
        result2 = " ".join(mylist)
    question2.append(result2)
    i = i+1


# In[40]:


# cleaning data
def my_tokenize(s):
    s = s.lower() # put into lower case form
    s = re.sub(r"[\'\"\?.()/:,]", ' ',s) # remove certain punctuations and replace it with space
    s = re.sub(r"\s+",' ',s) # remove extra space
    s = re.sub(r"not\s+",'not_', s) # deal with negation
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    porter = PorterStemmer() 
    tokens = [porter.stem(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in set(stopword)] # remove stopwords
    return tokens


# In[41]:


myquestion1 = []
myquestion2 = []
for i in range(len(df)):
    myresult1 = my_tokenize(df['question1'][i])
    myresult2 = my_tokenize(df['question2'][i])
    myquestion1.append(myresult1)
    myquestion2.append(myresult2)
    i = i+1


# In[48]:


df['question1']= myquestion1
df['question2']= myquestion2


# In[49]:


df


# In[51]:


import csv

with open('newtrain.csv', 'wb') as myfile:
    df.to_csv('newtrain.csv',  encoding='utf-8')


# ## Report
# 
# For data cleaning part, we mainly did following things:
# 
# Firstly, we find all empty questions, and make them equal to a string "wrong".
# 
# Then, we create our own list of stopwords and combined it with the stopwords list from nltk.corpus, but we choose to keep interrogative words. Because for example, if we have two questions: "Where is the book?" and "How is the book?", these are two completely different questions, then if we remove interrogative words, there is no way to distinguish them. We also discovered that many interrogative words are in contraction format, so in order to keep them, we found a dictionary and use it to spread out the contraction form and thus keep them not been removed by stopword list. 
# 
# Next, we create a function to clean the text and only keep important words. We first put words in to lower case and  remove certain symbols and punctuations and replace it with space, then we remove extra space. We also need to deal with negation problem, because we have dealt with contraction words, so we just need to concadinate "not" with next word so that it won't be removed by stopwords. Then we split the string into words and stem each words and remove stopwords, at last it returns a list of words for each questionm.
# 
