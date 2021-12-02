# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 11:17:44 2021

@author: ken
"""
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import fastai.text.all as ft

model =ft.load_learner('C:/Users/ken/Downloads/fine_tuned.pkl')

data = pd.read_csv('C:/Users/ken/Downloads/test.csv', delimiter=',')
docs_test = data['Review']
predict_result = []
for i in range(len(docs_test)):
    predict_result.append(model.predict(docs_test[i])[0])

predict_result = pd.Series(predict_result).astype(int)

accuracy = np.mean(predict_result == data['RatingValue'])
f1_score = f1_score(data['RatingValue'], predict_result, average='macro')
confusion = confusion_matrix(data['RatingValue'], predict_result)
cof_columns = ('negative', 'neutral', 'positive')
confusion_df = pd.DataFrame(confusion, columns = cof_columns, index = cof_columns)
print('accuracy: ', accuracy, '\nF1_score: ', f1_score, '\nConfusion_matrix: \n', confusion_df) 
