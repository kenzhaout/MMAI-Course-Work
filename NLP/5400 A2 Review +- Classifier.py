# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:37:31 2021

@author: ken
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


col_names=['CompanyName', 'RatingValue', 'DatePublished', 'ReviewBody'] 
data = pd.read_csv('C:/Users/ken/Downloads/reviews.csv', names = col_names, header = None, delimiter='\t')
data = data.drop_duplicates()

ratings = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
new_ratings = [ratings[item] for item in data['RatingValue']]
data['RatingValue'] = new_ratings

data['RatingValue'].value_counts()

df = data.groupby('RatingValue', as_index=False).apply(lambda x: x.sample(n=51, random_state = 1)).reset_index(drop=True)

result_table = pd.DataFrame({'Sentiment': df['RatingValue'], 'Review': df['ReviewBody']})
X_train, X_test , Y_train, Y_test = train_test_split(result_table['Review'],
                                                     result_table['Sentiment'],
                                                     test_size=0.3,
                                                     random_state=10)
train_df = pd.DataFrame(data = [Y_train, X_train]).T
test_df = pd.DataFrame(data = [Y_test, X_test]).T

#Output
train_df.to_csv('training.csv', index=False)
test_df.to_csv('valid.csv', index=False)

#Load train_data and build model
train = pd.read_csv('training.csv')

test_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=10,
                          max_iter=5, tol=None)),
])

test_clf.fit(train['Review'], train['Sentiment'])

#Load valid data and check performance
valid = pd.read_csv('valid.csv')
docs_test = valid['Review']
predicted = test_clf.predict(docs_test)
accuracy = np.mean(predicted == valid['Sentiment'])
f1_score = f1_score(valid['Sentiment'], predicted, average='macro')
confusion = confusion_matrix(valid['Sentiment'], predicted)
cof_columns = ('negative', 'neutral', 'positive')
confusion_df = pd.DataFrame(confusion, columns = cof_columns, index = cof_columns)

print('accuracy: ', accuracy, '\nF1_score: ', f1_score, '\nConfusion_matrix: \n', confusion_df) 
