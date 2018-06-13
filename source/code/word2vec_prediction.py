import pandas as pd
import gensim
import logging
import time
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/esmondchu/Dropbox/UC_Davis/STAT/STA141CSpring18/Final_Project/train.csv")

#################################
#########Data Cleaning###########
#################################

#remove punctuation
def string_process(str_in):
    """
    Change a sentence in to lowercase and remove punctuation
    :param str_in (str): the process sentence
    :return: str: the processed sentence
    """
    punc = '?,!.()\'":'
    str_out = str_in.lower()
    for i in punc:
        str_out = str_out.replace(i, " ")
    return str_out

data_question = data[['question1','question2','is_duplicate']]

#find problematic rows
drop_rows = []
for i in range(len(data_question.question2)):
    if type(data_question.question2[i]) == float:
        drop_rows.append(i)

for i in range(len(data_question.question1)):
    if type(data_question.question1[i]) == float:
        drop_rows.append(i)

#create a new copy for modification
new_data_df = data_question.copy()

#drop problematic rows
new_data_df.drop(new_data_df.index[drop_rows], inplace=True)

#remove punctuation
new_data_df.question1 = new_data_df.question1.apply(string_process)
new_data_df.question2 = new_data_df.question2.apply(string_process)

#split words in a sentence (also use this for word2vec section)
que1 = [i.split() for i in new_data_df.question1]
que2 = [i.split() for i in new_data_df.question2]

#build sentence base, question 1 combines with question 2
que = que1 + que2

#################################
###########Build Model###########
#################################
start = time.time()
model = gensim.models.Word2Vec(que, size=100, window=5, min_count=5, workers=4)
end = time.time()
print('Run Time:', end-start)

#################################
########Get predicted y##########
#################################
from progressbar import ProgressBar
bar = ProgressBar()

similarity_rate = 0.1
top_pick_num = 10

overlap_score_model = []
for q1, q2 in bar(zip(que1,que2)):
    score1 = 0
    score2 = 0
    #handle score 1
    for i in q1:
        try:
            check_list_q1 = model.most_similar(positive=[i])
            picked_q1 = [i[0] for i in check_list_q1 if i[1] >= similarity_rate]
            if len(picked_q1) <= top_pick_num:
                selected_q1 = picked_q1
            else:
                selected_q1 = picked_q1[0:top_pick_num]
            for i in selected_q1:
                if i in q2:
                    score1 += 1
        except:
            score1 = 0
    #handle score 2
    for i in q2:
        try:
            check_list_q2 = model.most_similar(positive=[i])
            picked_q2 = [i[0] for i in check_list_q2 if i[1] >= similarity_rate]
            if len(picked_q2) <= top_pick_num:
                selected_q2 = picked_q2
            else:
                selected_q2 = picked_q2[0:top_pick_num]
            for i in selected_q2:
                if i in q1:
                    score2 += 1
        except:
            score2 = 0

    overlapping_model = (score1 + score2)/(len(q1) + len(q2))
    overlap_score_model.append(overlapping_model)

#################################
#######Calculate Accuracy########
#################################
def cal_accuracy_model(thr):
    predicted_model = list(((np.array(overlap_score_model) - thr) > 0) * 1)
    accuracy_model = np.sum(predicted_model == new_data_df.is_duplicate)/len(predicted_model)
    return accuracy_model

#get accuracy (testing different thresholds)
accuracy_thr_model = [cal_accuracy_model(i) for i in list(np.arange(0,1,0.1))]

#################################
###Plot Accuracy vs Threshold####
#################################

x = list(np.arange(0,1,0.1))
plt.figure(figsize=(10,5))
plt.plot(x, accuracy_thr_model)
plt.title('Accuracy vs Threshold with word2vec implementation')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.grid()
plt.show()
