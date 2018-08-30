# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
import numpy as np

def main():
  predictionService()
  
def predictionService():
  # Load dataset
  df = pandas.read_csv("dataset/test.csv")
  print(df["workbook"].nunique())
  print("rows x cols {}".format(df.shape))
  df = df.drop(['year'], axis=1) # almost a constant
  df = df.drop(['minute'], axis=1) # We definitely do not need second and micro second level granularity and probably don't need minute as well
  numOfColumns = len(df.columns)
  #df.to_csv("in.csv") 
  # all values must be integers for modelling
  le = preprocessing.LabelEncoder()
  df["site"] = le.fit_transform(df["site"])
  
  df = normalize(df) #optional normalization
  #df.to_csv("out.csv")  
  array = df.values
  X = array[:,0:numOfColumns-1] # all features
  Y = array[:,numOfColumns-1] # output
  validation_size = 0.01
  seed = 0
  X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

  #spotCheck(X_train,Y_train, X_validation, Y_validation, seed)
  
  predict_knn(X_train,Y_train, X_validation, Y_validation)
  #predict_CART(X_train,Y_train, X_validation, Y_validation)


def predict_knn(X_train,Y_train, X_validation, Y_validation):
  print("knn prediction begin")
  knn = KNeighborsClassifier(n_neighbors=5, p=2, weights='distance', n_jobs=-1, algorithm='kd_tree',leaf_size=2)
  knn.fit(X_train, Y_train)
  print("knn training complete")
  predictions = knn.predict(X_validation)
  print("Accuracy : %.2f %%" % (100*accuracy_score(Y_validation, predictions)))
  #print(confusion_matrix(Y_validation, predictions))
  #print(classification_report(Y_validation, predictions))
  print("knn prediction end")

def predict_CART(X_train,Y_train, X_validation, Y_validation):
  print("knn prediction begin")
  knn = KNeighborsClassifier(n_neighbors=5, p=2, weights='distance', n_jobs=-1, algorithm='kd_tree',leaf_size=2)
  multi_target_forest = MultiOutputClassifier(knn, n_jobs=-1)
  multi_target_forest.fit(X_train, Y_train)
  predictions = multi_target_forest.predict(X_validation)
  print("Accuracy : %.2f %%" % (100*accuracy_score(Y_validation, predictions)))
  #print(confusion_matrix(Y_validation, predictions))
  #print(classification_report(Y_validation, predictions))
  print("knn prediction end")

def spotCheck(X_train,Y_train, X_validation, Y_validation, seed):
  print("spot check Begin")
  # Spot Check Algorithms
  scoring = 'accuracy'
  models = []
  #models.append(('LR', LogisticRegression()))
  #models.append(('LDA', LinearDiscriminantAnalysis()))
  models.append(('KNN', KNeighborsClassifier()))
  #models.append(('CART', DecisionTreeClassifier()))
  #models.append(('NB', GaussianNB()))
  #models.append(('SVM', SVC()))
  # evaluate each model in turn
  results = []
  names = []
  for name, model in models:
    kfold = model_selection.KFold(n_splits=3, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring, n_jobs=-1, verbose=0)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
  print("spot check End")
  
def normalize(df):
  dfOrig = df;
  df = df.drop(['workbook'], axis=1)
  min_max_scaler = preprocessing.MinMaxScaler()
  arr_scaled = min_max_scaler.fit_transform(df)
  df = pandas.DataFrame(arr_scaled, columns=list(df));
  df["workbook"] = dfOrig["workbook"]
  return df
  

def normalize_old(df):
  # normalize all features except the "to be predicted" feature
  dfOrig = df;
  arr = df.drop(['workbook'], axis=1).values
  min_max_scaler = preprocessing.MinMaxScaler()
  arr_scaled = min_max_scaler.fit_transform(arr);
  df = pandas.DataFrame(arr_scaled)
  df["workbook"] = dfOrig["workbook"]
  return df
  
if __name__ == '__main__':
  main()
