
# coding: utf-8

# In[72]:


from numpy import loadtxt
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


# In[73]:


# load data
dataset = loadtxt('/home/venky/Downloads/reformatted.csv', delimiter=",")
np.random.shuffle(dataset)
print len(dataset)

# In[57]:


# split data into X and y
X = dataset[:,0:3]
Y = dataset[:,3]
np.delete(dataset,[0,1,2,3])


# In[58]:


# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# In[62]:


# fit model no training data
model = XGBClassifier(n_estimators = 1500,max_depth = 8)
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[63]:


print(precision_recall_fscore_support(y_test, predictions, average='macro'))


# In[64]:


from xgboost import plot_tree
import matplotlib.pyplot as plt

plot_tree(model)
plt.show()


# In[68]:


# In[71]:


print len(dataset)

