import pandas as pd
import numpy as np
import sklearn
from sklearn import metrics
from sklearn import preprocessing
    
#from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

import sys
if sys.version_info >= (3, 0):
    print("Error: You are running Python 3.x. This pynb is written in Python 2.")


# Load data
bc = pd.read_csv('companies_imputed_simplified.csv', header=0)

"""
header_labels = []
with open("field_names.txt") as header_labels_file:
    print "Column Headers:\n"
    for line in header_labels_file:
        header_labels.append(line.strip())
for label in header_labels:
    print label
print "\nNumber of labels loaded: %.f" %len(header_labels)

bc.columns=header_labels
"""

bc=bc.sample(frac=1,random_state=5).reset_index(drop=True)
bc_Y = pd.get_dummies(bc.ipo_is_true)
print "bc_Y.head():"
print bc_Y.head()
bc_Y.columns = ['ipo_FALSE','ipo_TRUE']
bc_Y = bc_Y.drop('ipo_FALSE',1)
#bc_Y.columns = ['ipo_TRUE','ipo_TRUE']
print "bc_Y.head():"
print bc_Y.head()


bc_X = bc.drop('ipo_is_true', 1).drop('ID', 1)

#print "Dataframe shape: %.f rows, %.f columns\n" % bc_Y.shape
#print "Check first few rows of bc_Y (outcome variable):"
#print bc_Y.head(5)
#print "Dataframe shape (feature vars): %.f rows, %.f columns\n" % bc_X.shape
#print "Check first few rows of  of bc_X (input variables):"
#bc_X.head(5)

#Data Setup
bc_cols = bc_X.columns
X = bc_X.values #returns a numpy array
#min_max_scaler = preprocessing.MinMaxScaler()
#X_scaled = min_max_scaler.fit_transform(X)
#X_scaled = pd.DataFrame(X_scaled)
#X_scaled.columns = bc_cols
X_scaled = bc_X



#Splitting Test/Train Data
train_data,train_labels = X_scaled[:450],bc_Y[:450]
test_data,test_labels = X_scaled[450:],bc_Y[450:]
print "Test data, IPO True ratio [%.2f]:\t%.0f out of %.0f" % (float(
        test_labels[test_labels.ipo_TRUE==1].shape[0]/1./test_labels.shape[0]),
                                                                test_labels[test_labels.ipo_TRUE==1].shape[0],
                                                                test_labels.shape[0])
print "Training data, IPO True ratio [%.2f]:\t%.0f out of %.0f" % (float(
        train_labels[train_labels.ipo_TRUE==1].shape[0]/1./train_labels.shape[0]),
                                                                    train_labels[train_labels.ipo_TRUE==1].shape[0],
                                                                    train_labels.shape[0])

#bc_NewModel = GaussianNB()
bc_NewModel = DecisionTreeClassifier()

print "\n\n\n\t\t=== DEBUGGING ==="
print type(train_labels.values[6])
#print type(train_labels.values.ravel())
#print type(train_data.values.ravel())
#print type(test_data.values)
#print "\n\n\n\t\t=== END DEBUGGING ==="

print "MODEL FITTing"
bc_NewModel.fit(train_data.values,train_labels.values.ravel())
print "MODEL FITTED"
model_predictions = bc_NewModel.predict(test_data.values)


print "\n"
print bc_NewModel
print "SKLearn calc accuracy:\t\t\t%.2f%%" % float(100*sklearn.metrics.accuracy_score(model_predictions,test_labels))
print "SKLearn calc accuracy (IPO False):\t%.2f%%" % float(100*sklearn.metrics.accuracy_score(model_predictions[test_labels.ipo_TRUE.values==0],test_labels[test_labels.ipo_TRUE.values==0]))
print "SKLearn calc accuracy (IPO True):\t%.2f%%" % float(100*sklearn.metrics.accuracy_score(model_predictions[test_labels.ipo_TRUE.values==1],test_labels[test_labels.ipo_TRUE.values==1]))
print "SKLearn calc f1 score:\t%.2f" % sklearn.metrics.f1_score(model_predictions,test_labels)
print "\n"

#for reference: from sklearn.tree import export_graphviz
export_graphviz(bc_NewModel,feature_names=test_data.columns)

print train_data.columns

