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

  print("Training   time for %10d samples       : %d seconds (Rate %d samples/second)" % (Y_train.size, trainTime, Y_train.size/trainTime))
  modelFile = "knn.model-" + sys.argv[1]
  print("Saving model to " +  modelFile + " file\n")
  pickle.dump(knnModel, open(modelFile, 'wb'))
  uniqueWorkbooksSorted = sorted(df["workbook"].unique())
  pickle.dump(uniqueWorkbooksSorted, open("uniqueWorkbooksSorted", 'wb'))

def predict():
  df = loadData()
  avgUniqueWorkbooksPerSite = df["workbook"].nunique() * 1.0/df["site"].nunique()
  randomAccuracy = 100.0/avgUniqueWorkbooksPerSite
  df = getSiteEncodings(df)
  [df, X_validation, Y_validation] = split(df)
  modelFile = "knn.model-" + sys.argv[4]
  print("Using " + modelFile + " file")
  knnModel = pickle.load(open(modelFile, 'rb'))

  startTime = time.time()
  predictions = knnModel.predict(X_validation)
  predictionTime = time.time() - startTime
  print("Random prediction Accuracy                   : %.1f %%" % randomAccuracy)
  print("KNN prediction Accuracy                      : %.1f %%" % (100*accuracy_score(Y_validation, predictions)))
  print("Prediction time for %10d samples       : %d seconds (Rate %d samples/second)\n" %
  (Y_validation.size, predictionTime, Y_validation.size/predictionTime))

  # get the probabilities of each predicted value
  probabilities = knnModel.predict_proba(X_validation)
  # choose top K probable values and store their indices
  K = 5
  topKValuesIndices = np.empty((Y_validation.size,K))
  for i in range(0, Y_validation.size):
    topKValuesIndices[i] = probabilities[i].argsort()[-K:]

  # read all possible prediction values from file which was created during training phase
  uniqueWorkbooksSorted = pandas.read_pickle("uniqueWorkbooksSorted")

  # use the prediction values domain to map indices to actual prediction values
  topKPredictions = np.empty((Y_validation.size,K),dtype='object')
  for i in range(0, Y_validation.size):
    for j in range(0, K):
      topKPredictions[i][j] = uniqueWorkbooksSorted[int(topKValuesIndices[i][j])]

  correctPredictions=0
  for i in range(0, Y_validation.size):
    if(Y_validation[i] in topKPredictions[i]):
      correctPredictions = correctPredictions + 1
  accuracy = correctPredictions*100.0/Y_validation.size
  print("KNN prediction Accuracy with %d predictions   : %.1f %%" % (K, accuracy))

def loadData():
  input = "dataset/" + sys.argv[1] + ".csv"
  df = pandas.read_csv(input)
  df = df.drop(['year'], axis=1) # almost a constant
  df = df.drop(['minute'], axis=1) # We definitely do not need second and micro second level granularity and probably don't need minute as well
  if(sys.argv[2] == "predict"):
    weekStartDate = df['day'].min()
    offset = int(sys.argv[3])-1
    actualDate = weekStartDate+offset
    df = df[df['day']==actualDate]
  numOfColumns = len(df.columns)
  numOfRows = len(df)

  print("Features:")
  for i in range(0,numOfColumns-1):
    print(df.columns[i], end='\t')
  print("\n\nTarget to predict: \n%s\n" % df.columns[numOfColumns-1])

  print("# of records                                 : %d" % numOfRows)
  print("# of unique sites                            : %d" % df["site"].nunique())
  print("# of unique workbooks                        : %d (Avg %.1f workbooks/site)" % (df["workbook"].nunique(), df["workbook"].nunique() * 1.0/df["site"].nunique()))
  return df

def encodeSite(df):
  # all values must be integers for modelling
  df.dropna(inplace=True)
  dfToSave = df["site"].copy().to_frame()
  le = preprocessing.LabelEncoder()
  df["site"] = le.fit_transform(df["site"])
  # use a large scaling factor so that even if the times are identical, two workbooks from different sites are far apart in the k-dimensional space tree
  df["site"] = 30.42*23.0*30.42*23.0*df["site"]
  dfToSave["encodedsite"] = df["site"].copy()
  dfToSave.drop_duplicates(inplace=True)
  dfToSave.reset_index(drop=True, inplace=True)
  siteEncodingsFile = "siteEncodings-" + sys.argv[1]
  dfToSave.to_pickle(siteEncodingsFile)
  return df

def getSiteEncodings(df):
  siteEncodingsFile = "siteEncodings-" + sys.argv[4]
  siteEncodings = pandas.read_pickle(siteEncodingsFile)
  siteEncodings = siteEncodings.set_index("site")["encodedsite"].to_dict()
  df["site"] = df.apply(lambda df : siteEncodings.get(df["site"]), axis=1)
  df.dropna(inplace=True)
  df.reset_index(drop=True, inplace=True)
  return df

def split(df):
  df = scale(df) #weighted scaling
  numOfColumns = len(df.columns)
  array = df.values
  X = array[:,0:numOfColumns-1] # all features
  Y = array[:,numOfColumns-1] # output
  return [df, X, Y]

def scale(df):
  # scale "hour" feature to vary between 0 and 1
  df["hour"] = df["hour"]/23.0
  # same hour from day i and day (i+1) should be apart by 23 hours in the kd tree
  df["day"] = 23.0*df["day"]/31.0
  df["month"] = 30.42*23.0*df["month"]/12.0
  return df

if __name__ == '__main__':
  main()
