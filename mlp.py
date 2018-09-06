# Load libraries
import time
import pandas
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import numpy as np
import pickle
import sys

def main():
  predictionService()
  
def predictionService():
  # Load dataset
  df = pandas.read_csv("dataset/test.csv")
  print("# of unique workbooks:", df["workbook"].nunique())
  print("Random prediction Accuracy : %.4f %%" % (100/(df["workbook"].nunique())))
  print("rows x cols {}\n".format(df.shape))
  df = df.drop(['year'], axis=1) # almost a constant
  df = df.drop(['minute'], axis=1) # We definitely do not need second and micro second level granularity and probably don't need minute as well
  numOfColumns = len(df.columns)
  numOfRows = len(df)

  # all values must be integers for modelling
  le = preprocessing.LabelEncoder()
  df["site"] = le.fit_transform(df["site"])
  
  df = normalize(df) #optional normalization
  array = df.values
  X = array[:,0:numOfColumns-1] # all features
  Y = array[:,numOfColumns-1] # output
  validation_size = 0.01
  
  X_train = X[0:int((1-validation_size)*numOfRows),:]
  X_validation = X[int((1-validation_size)*numOfRows):,:]
  Y_train = Y[0:int((1-validation_size)*numOfRows)]
  Y_validation = Y[int((1-validation_size)*numOfRows):]
  assert numOfRows == len(X_train) + len(X_validation)
  print("mlp begin")
  
  if(len(sys.argv) < 2):
    train_predict_mlp(X_train,Y_train, X_validation, Y_validation)
    return

  if(sys.argv[1] == "t"):  
    train_mlp(X_train,Y_train, X_validation, Y_validation)
  elif(sys.argv[1] == "p"):
    predict_mlp(X_validation, Y_validation)
  else:
    train_predict_mlp(X_train,Y_train, X_validation, Y_validation)

def train_mlp(X_train,Y_train, X_validation, Y_validation):
  startTime = time.time()
  mlpModel = MLPClassifier(hidden_layer_sizes=(5,), max_iter=2, alpha=0.0001, solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
  mlpModel.fit(X_train, Y_train)
  trainTime = time.time() - startTime
  print("mlp training complete")
  pickle.dump(mlpModel, open("mlp.model", 'wb'))
  print("Training   time for %d samples : %d seconds (Rate %d samples/second)" % (Y_train.size, trainTime, Y_train.size/trainTime))
  
def predict_mlp(X_validation, Y_validation):
  mlpModel = pickle.load(open("mlp.model", 'rb'))
  startTime = time.time()
  predictions = mlpModel.predict(X_validation)
  predictionTime = time.time() - startTime
  print("Accuracy : %.2f %%" % (100*accuracy_score(Y_validation, predictions)))
  print("mlp prediction done\n")
  print("Prediction time for %d samples : %d seconds (Rate %d samples/second)" % (Y_validation.size, predictionTime, Y_validation.size/predictionTime))
  
def train_predict_mlp(X_train,Y_train, X_validation, Y_validation):
  startTime = time.time()
  mlpModel =  MLPClassifier(hidden_layer_sizes=(5,), max_iter=2, alpha=0.0001, solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
  mlpModel.fit(X_train, Y_train)
  trainTime = time.time() - startTime
  print("mlp training complete")
  startTime = time.time()
  predictions = mlpModel.predict(X_validation)
  predictionTime = time.time() - startTime
  print("Accuracy : %.2f %%" % (100*accuracy_score(Y_validation, predictions)))
  print("mlp prediction done\n")
  print("Training   time for %d samples : %d seconds (Rate %d samples/second)" % (Y_train.size, trainTime, Y_train.size/trainTime))
  print("Prediction time for %d samples : %d seconds (Rate %d samples/second)" % (Y_validation.size, predictionTime, Y_validation.size/predictionTime))

 
def normalize(df):
  dfOrig = df;
  df = df.drop(['workbook'], axis=1)
  min_max_scaler = preprocessing.MinMaxScaler()
  arr_scaled = min_max_scaler.fit_transform(df)
  df = pandas.DataFrame(arr_scaled, columns=list(df));
  df["workbook"] = dfOrig["workbook"]
  return df
  
if __name__ == '__main__':
  main()
