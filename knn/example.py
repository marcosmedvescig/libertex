from knn_predictor import knn_predict
import random
import seaborn as sns
import pandas as pd

# create train sample
class_0 = [(random.randrange(0, 100), random.randrange(0, 100)) for i in range(100)]
class_1 = [(random.randrange(100, 200), random.randrange(100, 200)) for i in range(100)]

x_train = class_0 + class_1
y_train = [0] * 100 + [1] * 100

data = pd.DataFrame(x_train, columns =['x', 'y'])
data['class_train'] = y_train

# create test sample
x_test = [(random.randrange(0, 200), random.randrange(0, 200)) for i in range(30)]

# predict for test sample
predictions = pd.DataFrame(x_test, columns =['x', 'y'])
predictions['class_predict'] = knn_predict(x_train,y_train,x_test,k=13)

# plot results
sns.scatterplot(data = data, x= 'x', y='y',hue='class_train',palette='Set2',alpha=0.5)
sns.scatterplot(data = predictions, x= 'x', y='y',hue='class_predict',palette='Set2');
