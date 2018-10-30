# Load libraries
from __future__ import print_function
import time
import pandas
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import numpy as np
import pickle
import sys

def main():
  predictionService()

def predictionService():
  if(sys.argv[2] == "train"):
    train()
  elif(sys.argv[2] == "predict"):
    predict()

def train():
  df = loadData()
  df = encodeSite(df)
  [df, X_train, Y_train] = split(df)

  startTime = time.time()
  knnModel = KNeighborsClassifier(n_neighbors=6, p=2, weights='distance', n_jobs=-1, algorithm='kd_tree',leaf_size=30)
  knnModel.fit(X_train, Y_train)
  trainTime = time.time() - startTime

  print("Training   time for %10d samples : %d seconds (Rate %d samples/second)" % (Y_train.size, trainTime, Y_train.size/trainTime))
  print("Saving model to knn.model file\n")
  pickle.dump(knnModel, open("knn.model", 'wb'))

def predict():
  print("Loading knn.model file")
  df = loadData()
  df = getSiteEncodings(df)
  [df, X_validation, Y_validation] = split(df)
  knnModel = pickle.load(open("knn.model", 'rb'))

  startTime = time.time()
  predictions = knnModel.predict(X_validation)
  predictionTime = time.time() - startTime

  print("Random prediction Accuracy             : %.4f %%" % (100.0/(df["workbook"].nunique())))
  print("KNN prediction Accuracy                : %.2f %%" % (100*accuracy_score(Y_validation, predictions)))
  print("Prediction time for %10d samples : %d seconds (Rate %d samples/second)\n" % (Y_validation.size, predictionTime, Y_validation.size/predictionTime))

def loadData():
  input = "dataset/" + sys.argv[1] + ".csv"
  df = pandas.read_csv(input)
  df = df.drop(['year'], axis=1) # almost a constant
  df = df.drop(['minute'], axis=1) # We definitely do not need second and micro second level granularity and probably don't need minute as well
  numOfColumns = len(df.columns)
  numOfRows = len(df)

  print("Features:")
  for i in range(0,numOfColumns-1):
    print(df.columns[i], end='\t')
  print("\n\nTarget to predict: \n%s\n" % df.columns[numOfColumns-1])

  print("# of records                           : %d" % numOfRows)
  print("# of unique sites                      : %d" % df["site"].nunique())
  print("# of unique workbooks                  : %d (Avg %.1f workbooks/site)" % (df["workbook"].nunique(), df["workbook"].nunique() * 1.0/df["site"].nunique()))
  return df

def encodeSite(df):
  # all values must be integers for modelling
  dfToSave = df["site"].copy().to_frame()
  le = preprocessing.LabelEncoder()
  df["site"] = le.fit_transform(df["site"])
  dfToSave["encodedsite"] = df["site"].copy()
  dfToSave.drop_duplicates(inplace=True)
  dfToSave.reset_index(drop=True, inplace=True)
  dfToSave.to_pickle("siteEncodings")
  return df
  
def getSiteEncodings(df):
  siteEncodings = pandas.read_pickle("siteEncodings")
  siteEncodings = siteEncodings.set_index("site")["encodedsite"].to_dict()
  df["site"] = df.apply(lambda df : siteEncodings.get(df["site"]), axis=1)
  df.dropna(inplace=True)
  df.reset_index(drop=True, inplace=True)
  return df
  
def split(df):
  df = normalize(df) #optional normalization
  numOfColumns = len(df.columns)
  array = df.values
  X = array[:,0:numOfColumns-1] # all features
  Y = array[:,numOfColumns-1] # output
  return [df, X, Y]

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
