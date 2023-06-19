## LOGISTIC REGRESSION
## REVIEW CLASSIFICATION BASED ON WORDS

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
 
vectorizer = CountVectorizer()

## Fit the bag-of-words model
bag = vectorizer.fit_transform(women_clothes_reviews['Final Text'])

# Creating training data set from bag-of-words and dummy label
X = bag.toarray()
y = np.array(women_clothes_reviews['Recommended IND'])
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Create training and test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create an instance of LogisticRegression classifier
lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')

# Fit the model
model=lr.fit(X_train, y_train)

# Create the predictions
y_predict = model.predict(X_test)
y_predict_prob = model.predict_proba(X_test)
  
# Use metrics.accuracy_score to measure the score
print("LogisticRegression Accuracy %.3f" %metrics.accuracy_score(y_test, y_predict))

#################################
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_predict)

## Self-defined threshold
y_predict_self = [1 if prob > 0.102232 else 0 for prob in y_predict_prob[:,1]]
# Model evaluation on accuracy
confusion_matrix(y_test,y_predict_self)


from sklearn.metrics import roc_curve
from matplotlib import pyplot

fpr, tpr, thresholds = roc_curve(y_test, y_predict_prob[:,1])
# plot the roc curve for the model
pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
# show the plot
pyplot.show()

## Optimal THRESHOLD BASED ON F1-SCORE 
from numpy import argmax
from sklearn.metrics import precision_recall_curve
# calculate roc curves
precision, recall, thresholds = precision_recall_curve(y_test, y_predict_prob[:,1])
# convert to f score
fscore = (2 * precision * recall) / (precision + recall)
# locate the index of the largest f score
ix = argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
