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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
                    
def main(argv):
  predictionService(argv)
  
def predictionService(argv):
  # Load dataset
  df = pandas.read_csv("dataset/test.csv")
  print("rows x cols {}".format(df.shape))
  
  df = df.drop(['year'], axis=1) # almost a constant
  df = df.drop(['minute'], axis=1) # We definitely do not need second and micro second level granularity and probably don't need minute as well
  numOfColumns = len(df.columns)
  # all values must be integers for modelling
  le = preprocessing.LabelEncoder()
  df["site"] = le.fit_transform(df["site"])
  df["workbook"] = le.fit_transform(df["workbook"])
  
  #df = normalize(df) #optional normalization
  
  array = df.values
  X = array[:,0:numOfColumns-1] # all features
  Y = array[:,numOfColumns-1] # output
  numOfClasses = np.amax(Y).item() + 1
  print("Number of classes {}".format(numOfClasses))
  validation_size = 0.20
  seed = 0
  X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
  X_train = pandas.DataFrame(X_train, columns=['month', 'day', 'hour', 'site'])
  X_validation = pandas.DataFrame(X_validation, columns=['month', 'day', 'hour', 'site'])
  Y_train = pandas.Series(Y_train, name = 'workbook', dtype=int)
  Y_validation = pandas.Series(Y_validation, name = 'workbook', dtype=int)
  predict(argv, X_train, Y_train, X_validation, Y_validation, numOfClasses)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset
    
def predict(argv, X_train,Y_train, X_validation, Y_validation, numOfClasses):
    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in X_train.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    args = parser.parse_args(argv[1:])
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200],
        # The model must choose between n classes.
        n_classes=numOfClasses)

    # Train the Model.
    classifier.train(input_fn=lambda:train_input_fn(X_train, Y_train, args.batch_size), steps=args.train_steps)

    # Evaluate the model.
    accuracy_score = classifier.evaluate(input_fn=lambda:eval_input_fn(X_validation, Y_validation, args.batch_size))["accuracy"]
    print("\nTest Accuracy: {0:0.2f}%\n".format(accuracy_score*100))
  
def normalize(df):
  # normalize all features except the "to be predicted" feature
  dfOrig = df;
  arr = df.drop(['workbook'], axis=1).values
  min_max_scaler = preprocessing.MinMaxScaler()
  arr_scaled = min_max_scaler.fit_transform(arr);
  df = pandas.DataFrame(arr_scaled)
  df["workbook"] = dfOrig["workbook"]
  return df
  
if __name__ == '__main__':
  tf.app.run(main)
