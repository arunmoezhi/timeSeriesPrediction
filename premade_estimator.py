#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifie"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pandas as pd
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

CSV_COLUMN_NAMES = ['month', 'day',
                    'hour', 'site', 'workbook']

def load_data(y_name='workbook'):
    """Returns the dataset as (train_x, train_y), (test_x, test_y)."""

    train = pd.read_csv("dataset/train0.csv", names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv("dataset/test0.csv", names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)
    return (train_x, train_y), (test_x, test_y)

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
    
def run0(argv):
    # Fetch the data
    (train_x, train_y), (test_x, test_y) = load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    print(type(train_x))
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    args = parser.parse_args(argv[1:])
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=8)

    # Train the Model.
    print(train_x)
    print("\n")
    print(train_y)
    print("\n")
    print(test_x)
    print("\n")
    print(test_y)
    classifier.train(input_fn=lambda:train_input_fn(train_x, train_y, args.batch_size), steps=args.train_steps)

    # Evaluate the model.
    accuracy_score = classifier.evaluate(input_fn=lambda:eval_input_fn(test_x, test_y, args.batch_size))["accuracy"]
    #print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**accuracy_score))
    print("\nTest Accuracy: {0:0.2f}%\n".format(accuracy_score*100))

def main(argv):
    run0(argv)
    
if __name__ == '__main__':
    #tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
