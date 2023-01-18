import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,roc_auc_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import seaborn as sns

class ML_Classification():

    def __init__(self):
        self.classifier = RandomForestClassifier()
        self.predicted = None
        self.features_num=X.shape[1]
        self.XTrainKF=None
        self.yTrainKF=None
        self.XValidKF=None
        self.yValidKF=None
        self.acc_log=None

    def decisionTree(self):
        self.classifier = DecisionTreeClassifier(max_depth=3)

    def knn(self):
        self.classifier = KNeighborsClassifier(n_neighbors=4)

    def randomForest(self):
        self.classifier = RandomForestClassifier()

    def train(self, X_train, y_train):
        X_train = scaler(X_train)
        self.classifier.fit(X_train, y_train)

    def test(self, X_test, y_test):  
        X_test = scaler(X_test)
        self.predicted = self.classifier.predict(X_test)
        acc = accuracy_score(self.predicted, y_test)
        
        plt.clf()
        cm = sns.heatmap(confusion_matrix(
            y_test, self.predicted), annot=True, cmap='Blues')
        plt.savefig('./temp/cm.png')
        return acc
    
    def cv(self,n_splits,X,y):
        self.XTrainKF=[]
        self.yTrainKF=[]
        self.XValidKF=[]
        self.yValidKF=[]
        self.acc_log=[]

        kf=model_selection.StratifiedKFold(n_splits)
        for fold,(trn_,val_) in enumerate(kf.split(X,y)):

             
            self.XTrainKF.append(X[trn_,0:self.features_num])
            self.yTrainKF.append(y[trn_])
            self.XValidKF.append(X[val_,0:self.features_num])
            self.yValidKF.append(y[val_])

            xTrain=scaler(self.XTrainKF[fold])
            yTrain=self.yTrainKF[fold]
            xValid=scaler(self.XValidKF[fold])
            yValid=self.yValidKF[fold]
            
            self.train(xTrain,yTrain)
            self.predicted = self.classifier.predict(xValid)
            acc=roc_auc_score(yValid,self.predicted )
            self.acc_log.append(acc)
            acc = accuracy_score(yValid,self.predicted )
            cm = sns.heatmap(confusion_matrix(
            yValid, self.predicted), annot=True, cmap='Blues')
            plt.clf()
            path='./temp/iterasyon/cm'+str(fold)+'.png'
            plt.savefig(path)     

def load_dataset(dataset):
    global X, y
    global X_train, y_train, X_test, y_test

    dataset = pd.read_csv("./datasets/heart.csv")
    datas = dataset.values

    global attr_num
    attr_num = datas.shape[1]
    y = datas[:, attr_num-1]
    X = datas[:, 0:attr_num-1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    return X, y, X_train, y_train, X_test, y_test, attr_num

def scaler(input_X):
    scaler = StandardScaler()
    output_X = scaler.fit_transform(input_X)
    return output_X

def train_model(model,cv,n="2"):
    models = ["Decision Tree", "Random Forest", "KNN"]
    global classification
    classification = ML_Classification()    
    try:
        if model == models[0]:
            classification.decisionTree()
        elif model == models[1]:
            classification.randomForest()
        elif model == models[2]:
            classification.knn()
        
        if cv:
            classification.cv(int(n),X_train,y_train)

        else:
            classification.train(X_train, y_train)

    except Exception as e: # work on python 3.x
        print('Failed to upload to ftp: '+ str(e))

def test_model():
        acc = classification.test(X_test, y_test)
        return acc

def save_model(path):
     try:
         model = classification.classifier
         joblib.dump(model, path)
         return True

     except:
         return False

def load_model(path):
    return joblib.load(path)

def folds():
    return classification.XTrainKF,classification.yTrainKF,classification.XValidKF,classification.yValidKF,classification.acc_log

