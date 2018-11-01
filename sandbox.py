import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np

def main():
	#labelEncoder()
	gettingStarted()

def gettingStarted():
	# Load dataset
	df = pandas.read_csv("test.csv")
	print("rows x cols {}".format(df.shape))
	dfOrig = df;
	#df = pandas.read_csv("jan01-07-full-cleaned.csv")
	print("rows x cols {}".format(df.shape))
	# head
	print(df.head(5))

	#stats
	print(df.describe())
	print(df.groupby('site').size())

	#histograms
	df.hist()
	plt.show()

	#scatter plot matrix
	scatter_matrix(df)
	plt.show()

	# split the dataset into train and validation set
	x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	print(x)
	print("\n")
	print(x[:,0:2])
	print("\n")
	print(x[:,2:])
	
def labelEncoder(): 	
	from sklearn.metrics import classification_report
	from sklearn import preprocessing
	x = 3.0/4.1
	print("Accuracy : %.2f %%" % (x*100))
	origY = ['c1','c3','c0','c0','c2']
	le = preprocessing.LabelEncoder()
	origYEncoded = le.fit_transform(origY)
	y_true = origYEncoded
	print(y_true)
	print(le.inverse_transform(y_true))
	y_pred = [1, 3, 0, 1, 2]
	print(classification_report(y_true, y_pred))
  
def featureHasher(df):
  # creates less number of features compared to one-hot encoding for a categorical nominal feature
  fh = FeatureHasher(n_features=30, input_type='string')
  hashed_features = fh.fit_transform(df["site"])
  hashed_features = hashed_features.toarray()
  df1 = pandas.concat([df.iloc[:,0:3],pandas.DataFrame(hashed_features)],axis=1)
  df1["workbook"] = df["workbook"]
  return df

if __name__ == '__main__':
	main()