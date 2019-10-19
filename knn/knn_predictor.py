import pandas as pd
import numpy as np
import sys

def knn_predict(x_train, y_train, x_test, k=3,result_type = 'class'):

  # Parameter validation
  if len(x_train) < k:
      print('ERROR: need to have at least {0} train samples'.format(k))
      sys.exit
  if len(x_train) != len(y_train):
      print('ERROR: x_train and y_train must be same size')
      sys.exit
  if k < 2:
      print('ERROR: k must be 2 or larger')
      sys.exit
  if len(x_test) < 1:
      print('ERROR: need at least 1 sample to predict'.format(k))
      sys.exit
  if result_type not in ['class','proba']:
      print('ERROR: result_type must be class or proba')
      sys.exit

  # Run Predictions
  y_pred_list = []

  data = pd.DataFrame(x_train, columns =['x', 'y'])
  data['class'] = y_train

  for test_touple in x_test:
    data['distance'] =  data[['x', 'y']].sub(np.array(test_touple)).pow(2).sum(1).pow(0.5)
    data.sort_values(by='distance',inplace=True)

    # Predict Class
    if result_type == 'class':
      y_pred = data[0:k]['class'].value_counts().idxmax()
      # print('Class x: {0} y:{1} pred:{2}'.format(test_touple[0],test_touple[1],y_pred))

    # Predict Class Probability
    if result_type == 'proba':
      y_pred_tmp = data[0:k]['class'].value_counts(normalize=True)
      if 0 not in y_pred_tmp.keys():
         y_pred = [0, y_pred_tmp[1]]
      elif 1 not in y_pred_tmp.keys():
         y_pred = [y_pred_tmp[0],1]
      else:
        y_pred = [y_pred_tmp[0], y_pred_tmp[1]]
      # print('Proba x: {0} y:{1} proba:{2}'.format(test_touple[0],test_touple[1],y_pred))

    y_pred_list.append(y_pred)

  # Return predictions as an array
  y_pred_list = np.asarray(y_pred_list)

  return y_pred_list
