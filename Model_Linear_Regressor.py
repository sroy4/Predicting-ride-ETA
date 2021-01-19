from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
import pandas as pd
import tensorflow as tf


def _read_data():
  training_frame=pd.read_csv('training_frame.csv')
  test_frame=pd.read_csv('test_frame.csv')
  training_features=['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN', 'passenger_count', 'pickup_hour', 'trip_duration_by_county','trip_duration_by_neighborhood','trip_duration_by_city']
  target=['trip_duration']
  x_train=training_frame[training_features].values
  y_train=training_frame[target].values
  x_test=test_frame[training_features].values
  y_test=test_frame[target].values
  return (x_train, y_train), (x_test, y_test)

def main(unused_argv):
  # Load dataset

  (x_train, y_train), (x_test,
                       y_test) = _read_data()

  # Scale data (training set) to 0 mean and unit standard deviation.
  scaler = preprocessing.StandardScaler()
  x_train = scaler.fit_transform(x_train)

  feature_columns = [
      tf.feature_column.numeric_column('x', shape=np.array(x_train).shape[1:])]
  regressor = tf.estimator.LinearRegressor(
      feature_columns=feature_columns)

  # Train.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': x_train}, y=y_train, batch_size=1, num_epochs=None, shuffle=True)
  regressor.train(input_fn=train_input_fn, steps=2000)

  # Predict.
  x_transformed = scaler.transform(x_test)
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': x_transformed}, y=y_test, num_epochs=1, shuffle=False)
  predictions = regressor.predict(input_fn=test_input_fn)
  y_predicted = np.array(list(p['predictions'] for p in predictions))
  y_predicted = y_predicted.reshape(np.array(y_test).shape)

  # Score with sklearn.
  score_sklearn = metrics.mean_squared_error(y_predicted, y_test)
  print('MSE (sklearn): {0:f}'.format(score_sklearn))

  # Score with tensorflow.
  scores = regressor.evaluate(input_fn=test_input_fn)
  print('MSE (tensorflow): {0:f}'.format(scores['average_loss']))


if __name__ == '__main__':
  tf.app.run()
