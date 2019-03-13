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
  elif(sys.argv[2] == "probOfWbAtTime"):
    probOfWbAtTime(float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6]),float(sys.argv[7]),sys.argv[8],sys.argv[9])

def train():
  df = loadData()
  [df, X_train, Y_train] = split(df)
  startTime = time.time()
  knnModel = KNeighborsClassifier(n_neighbors=6, p=2, weights='distance', n_jobs=-1, algorithm='kd_tree',leaf_size=30)
  knnModel.fit(X_train, Y_train)
  trainTime = time.time() - startTime

  print("Training   time for %10d samples       : %d seconds (Rate %d samples/second)" % (Y_train.size, trainTime, Y_train.size/trainTime))
  modelFile = "knn.model-" + sys.argv[1] + "-" + sys.argv[3]
  print("Saving model to " +  modelFile + " file\n")
  pickle.dump(knnModel, open(modelFile, 'wb'))
  uniqueWorkbooksSorted = sorted(df["workbook"].unique())
  uniqueWorkbooksSortedFile = "uniqueWorkbooksSorted-" + sys.argv[1] + "-" + sys.argv[3]
  pickle.dump(uniqueWorkbooksSorted, open(uniqueWorkbooksSortedFile, 'wb'))

def probOfWbAtTime(year, month, day, hour, minute, site, workbook):
  print("(year: %s), (month: %s), (day: %s), (hour: %s), (min: %s), (site: %s), (workbook: %s)" % (year, month, day, hour, min, site, workbook))
  input = "dataset/" + sys.argv[1] + ".csv"
  d = {'year': [year], 'month': [month], 'day': [day], 'hour': [hour], 'minute': [minute], 'site': [site], 'workbook': [workbook]}
  df = pandas.DataFrame(data=d)
  #df = pandas.read_csv(input)
  df = df.drop(['year'], axis=1) # almost a constant
  df = df.drop(['minute'], axis=1) # We definitely do not need second and micro second level granularity and probably don't need minute as well

  df = df[df['site']==site]
  df = df.drop(['site'], axis=1)

  [df, X_validation, Y_validation] = split(df)
  modelFile = "knn.model-" + sys.argv[1] + "-" + site
  knnModel = pickle.load(open(modelFile, 'rb'))
  probabilities = knnModel.predict_proba(X_validation)
  #print(probabilities)
  # read all possible prediction values from file which was created during training phase
  uniqueWorkbooksSortedFile = "uniqueWorkbooksSorted-" + sys.argv[1] + "-" + site
  uniqueWorkbooksSorted = pandas.read_pickle(uniqueWorkbooksSortedFile)
  for i in range(0,len(uniqueWorkbooksSorted)):
    if(uniqueWorkbooksSorted[i] == workbook):
      print("Access Probability: %.2f" % probabilities[0][i])
      return
  print(0.0)


def predict():
  df = loadData()
  randomAccuracy = 100.0/df["workbook"].nunique()
  [df, X_validation, Y_validation] = split(df)
  modelFile = "knn.model-" + sys.argv[4] + "-" + sys.argv[5]
  print("Using " + modelFile + " file")
  knnModel = pickle.load(open(modelFile, 'rb'))

  startTime = time.time()
  predictions = knnModel.predict(X_validation)
  predictionTime = time.time() - startTime
  print("Random prediction Accuracy                   : %.1f %%" % randomAccuracy)
  print("KNN prediction Accuracy                      : %.1f %%" % (100*accuracy_score(Y_validation, predictions)))
  print("Prediction time for %10d samples       : %d seconds (Rate %d samples/second)\n" %
  (Y_validation.size, predictionTime, Y_validation.size/predictionTime))

  if(df["workbook"].nunique() > 4):
    multipleWorkbooksPrediction(knnModel, X_validation, Y_validation, 5)

def multipleWorkbooksPrediction(knnModel, X_validation, Y_validation, K):
  # get the probabilities of each predicted value
  # this is an array of size (# of records to predict X # of unique workbooks in the training set
  probabilities = knnModel.predict_proba(X_validation)
  # choose top K probable values and store their indices
  # this is an array of size Y_validation.size X K
  topKValuesIndices = np.empty((Y_validation.size,K))
  # sort (descending) every row based on the probability value and choose top K values
  for i in range(0, Y_validation.size):
    topKValuesIndices[i] = probabilities[i].argsort()[-K:]

  # read all possible prediction values from file which was created during training phase
  uniqueWorkbooksSortedFile = "uniqueWorkbooksSorted-" + sys.argv[4] + "-" + sys.argv[5]
  uniqueWorkbooksSorted = pandas.read_pickle(uniqueWorkbooksSortedFile)

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

  trainOrPredict = sys.argv[2]
  if(trainOrPredict == "train" and len(sys.argv) > 3):
    print("site: %s" % sys.argv[3])
    df = df[df['site']==sys.argv[3]]
  if(trainOrPredict == "predict" and len(sys.argv) > 5):
    print("site: %s" % sys.argv[5])
    df = df[df['site']==sys.argv[5]]

  df = df.drop(['site'], axis=1)

  if(trainOrPredict == "predict"):
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
  print("# of unique workbooks                        : %d (Avg %.1f workbooks/site)" % (df["workbook"].nunique(), df["workbook"].nunique()))
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
