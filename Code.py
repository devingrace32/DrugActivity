import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedKFold, cross_val_score 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import sklearn.neural_network
from statistics import mean
from tensorflow import keras
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras import layers


##function to read compressed matrix into full matrix
def to_matrix(filename, booli):
    file1 = open(filename)
    lines = file1.readlines()
    num_cols = 100001
    num_rows =  len(lines)
    empty_mat = np.zeros((num_rows,num_cols))
    count = 0
    for line in lines:
        line = line.replace("\t", " ")
        line_list = line.split(" ")
        x=0
        if(booli == False):
            empty_mat[count,0] = line_list[0]
            x=1
        for i in range(x,len(line_list)-1):
            empty_mat[count,int(line_list[i])] = 1
        count+=1
    return empty_mat

mat = to_matrix("C:/Users/devin/OneDrive/Documents/CS484/HW2/TrainingData.txt", False)
y = mat[:,0]
x = mat[:,1:]

'''
#####BALANCE DATA###########
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from collections import Counter
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=42)
oversample = SMOTEENN(random_state=42)
x_smote, y_smote = oversample.fit_resample(x_train,y_train)
'''

#####MODEL CREATION#######
#Decision Tree
'''
model1 = DecisionTreeClassifier(criterion="gini",random_state=21)
model1.fit(x_train, y_train)
y_predict = model1.predict(x_test)
'''

#Weighted Decision Tree: 0.69
scores=[]
weights={0:1.0,1:10.0}
model2 = DecisionTreeClassifier(class_weight=weights)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
for train_i, test_i in skf.split(x,y):
    x_train_fold = x[train_i]
    x_test_fold =  x[test_i]
    y_train_fold = y[train_i]
    y_test_fold = y[test_i]
    model2.fit(x_train_fold,y_train_fold)
    y_pred = model2.predict(x_test_fold)
    scores.append(f1_score(y_test_fold, y_pred))
print(mean(scores))


#ANN
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
def ann_model():
    model3 = Sequential()
    model3.add(Dense(units=16, activation='relu', input_dim=100000))
    model3.add(Dense(units=8, activation='relu'))
    model3.add(Dense(units = 1, activation='sigmoid'))
    model3.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=["binary_accuracy"])
    return model3

estim =KerasClassifier(build_fn = ann_model, epochs=10, batch_size=8, verbose=0)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
results = cross_val_score(estim, x, y, cv=skf, scoring="f1")
print(mean(results))

'''
#Logistic Regression
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(class_weight=weights)
model4 = clf.fit(x_train, y_train)
y4_predict= model4.predict(x_test)

model4 = sklearn.neural_network.MLPClassifier()
model4.fit(x_train, y_train)
y4_predict = model4.predict(x_test)

#naive bayes
model2 = GaussianNB()
model2.fit(x_train,y_train)
y2_predict = model2.predict(x_test)
'''

#######ENTRY PREP###########
test = to_matrix("C:/Users/devin/OneDrive/Documents/CS484/HW2/TestingData.txt", True)
model2.fit(x,y)
test_pred = model2.predict(test[:,1:])
with open("C:/Users/devin/OneDrive/Documents/CS484/HW2/TestTree.txt", 'w') as tester:
    for row in test_pred:
        if(row < 0.5):
            tester.write("0\n")
        else:
            tester.write("1\n")

print("done")
from sklearn.preprocessing import LabelEncoder

'''
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
modelz= ann_model()

modelz.fit(x,y, batch_size =32, epochs=100, class_weight=weights)
predie = modelz.predict(test[:,1:])
for row in predie:
    if(row > 0.05):
        print("1")
    else:
        print("0")

'''
'''
###PCA experiment###
from sklearn.decomposition import PCA
pca=PCA(n_components=50, random_state=21)
pca.fit(x_train)
x_pca = pca.transform(x_train)

pca.fit(x_test)
x_test_pca = pca.transform(x_test)
'''


