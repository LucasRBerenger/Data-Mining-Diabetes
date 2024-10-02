import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv(">> SET FILE PATH HERE <</diabetes.csv", encoding="latin1")

features = data.iloc[:, 0:8].values  # Features include all but the last column
target = data.iloc[:, 8]  # The 'Outcome' column is the target

# 0.25 divides into 4 parts
train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.25, random_state=0)

def accuracy(confusion_matrix):
    total_correct, total = 0, 0
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[0])):
            if i == j:
                total_correct += confusion_matrix[i, j]
            total += confusion_matrix[i, j]
    return total_correct / total

decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(train_features, train_target)

predictions = decision_tree.predict(test_features)

tree_cm = confusion_matrix(predictions, test_target)
print('\nDECISION TREE\n')
print('Confusion Matrix:')
print(tree_cm, '\n')
print('Accuracy: ', accuracy(tree_cm) * 100, '%\n')

knn = KNeighborsClassifier()
knn.fit(train_features, train_target)

knn_predictions = knn.predict(test_features)

knn_cm = confusion_matrix(knn_predictions, test_target)
print('K-NEAREST NEIGHBORS\n')
print('Confusion Matrix:')
print(knn_cm, '\n')
print('Accuracy: ', accuracy(knn_cm) * 100, '%\n')

svm = SVC(random_state=0, kernel='linear')
svm.fit(train_features, train_target)

svm_predictions = svm.predict(test_features)

svm_cm = confusion_matrix(svm_predictions, test_target)
print('\nSUPPORT VECTOR MACHINES\n')
print('Confusion Matrix:')
print(svm_cm, '\n')
print('Accuracy: ', accuracy(svm_cm) * 100, '%\n')
