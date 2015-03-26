# -*- coding: utf-8 -*-

from __future__ import division

import sqlite3
import pandas
import numpy

#load train data
df = pandas.read_csv('/Users/YihanLi/Documents/project/train.csv', index_col=False, header=0)
df.head()

# Sort the dataset by the suit of each card then by the value
df = df.sort(['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5'], ascending=[True, True, True, True, True, True, True, True, True, True])

#Cont the number of occurance of each suit
df['s1_count'] = 0
df['s2_count'] = 0
df['s3_count'] = 0
df['s4_count'] = 0
df['abs12'] = 0
df['abs23'] = 0
df['abs34'] = 0
df['abs45'] = 0
df['abs51'] = 0

#Add value for the new features
for index, row in df.iterrows():    
    count_s1=0
    count_s2=0
    count_s3=0
    count_s4=0
    abs12=0
    abs23=0
    abs34=0
    abs45=0
    if row['S1'] == 1:
        count_s1+=1
    elif row['S1'] == 2:
        count_s2+=1
    elif row['S1'] == 3:
        count_s3+=1
    else:
        count_s4+=1

    if row['S2'] == 1:
        count_s1+=1
    elif row['S2'] == 2:
        count_s2+=1
    elif row['S2'] == 3:
        count_s3+=1
    else:
        count_s4+=1
    
    if row['S3'] == 1:
        count_s1+=1
    elif row['S3'] == 2:
        count_s2+=1
    elif row['S3'] == 3:
        count_s3+=1
    else:
        count_s4+=1

    if row['S4'] == 1:
        count_s1+=1
    elif row['S4'] == 2:
        count_s2+=1
    elif row['S4'] == 3:
        count_s3+=1
    else:
        count_s4+=1

    if row['S5'] == 1:
        count_s1+=1
    elif row['S5'] == 2:
        count_s2+=1
    elif row['S5'] == 3:
        count_s3+=1
    else:
        count_s4+=1
    
    row['abs12'] = abs(row['C1']-row['C2'])
    row['abs23'] = abs(row['C2']-row['C3'])
    row['abs34'] = abs(row['C3']-row['C4'])
    row['abs45'] = abs(row['C4']-row['C5'])
    row['abs51'] = abs(row['C5']-row['C1'])

    row['s1_count'] = count_s1
    row['s2_count'] = count_s2
    row['s3_count'] = count_s3
    row['s4_count'] = count_s4
    
    
# Cross validate 20% of the data each time
CROSS_VALIDATION_AMOUNT = .5




#Define response and explanatory series
response_df = df.hand

explanatory_features = [col for col in df.columns if col not in ['hand']]
explanatory_df = df[explanatory_features]

#
#holdout_num = round(len(df.index) * CROSS_VALIDATION_AMOUNT, 0)
#test_indices = numpy.random.choice(df.index, holdout_num, replace = False )
#train_indices = df.index[~df.index.isin(test_indices)]
#
#response_train = response_df.ix[train_indices,]
#explanatory_train = explanatory_df.ix[train_indices,]
#
#response_test = response_df.ix[test_indices,]
#explanatory_test = explanatory_df.ix[test_indices,]
#

#

#
#clf_extra = RandomForestClassifier(n_estimators=100, criterion = 'gini', max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 400)
#
#
##One instance score
#clf_extra.fit(explanatory_train, response_train)
#predicted_values = clf_extra.predict(explanatory_test)
#
#number_correct = len(response_test[response_test == predicted_values])
#total_in_test_set = len(response_test)
#accuracy = number_correct / total_in_test_set
#print accuracy* 100
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
clf_extra = RandomForestClassifier(n_estimators=100, criterion = 'entropy', max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 400)
#cross val score

scores = cross_val_score(clf_extra, explanatory_df, response_df, cv=10, scoring='accuracy', n_jobs = -1)
mean_accuracy = numpy.mean(scores) 
print mean_accuracy * 100
clf_extra.fit(explanatory_df, response_df)
clf_extra.feature_importances_


























#Test data
test_df = pandas.read_csv('/Users/YihanLi/Documents/project/test.csv', index_col=False, header=0)

test_df.head()
test_response_df = test_df.hand

test_explanatory_df = test_df[explanatory_features]


CROSS_VALIDATION_AMOUNT = .2
