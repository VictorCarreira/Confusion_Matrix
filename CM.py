#This program implement the confusion matrix 
#Author: Victor Carreira
###################################### THEORY ################################################
#    In the field of machine learning and specifically the problem of statistical classification, a confusion matrix, also known as an error matrix,[6] is a specific table layout that allows visualization of the performance of an algorithm, typically a supervised learning one (in unsupervised learning it is usually called a matching matrix).      
#    Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class (or vice versa).[2] The name stems from the fact that it makes it easy to see if the system is confusing two classes (i.e. commonly mislabeling one as another).
# - References:
#   [2] Powers, David M W (2011). "Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation" (PDF). Journal of Machine Learning Technologies. 2 (1): 37–63.
#   [6] Stehman, Stephen V. (1997). "Selecting and interpreting measures of thematic classification accuracy". Remote Sensing of Environment. 62 (1): 77–89. doi:10.1016/S0034-4257(97)00083-7.

##############################################################################################


################################ REQUIRED PACKAGES  #######################################
import numpy as np
import pylab as py
import scipy as sp
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from string import ascii_uppercase
from pandas import Series, DataFrame
from sklearn.metrics import confusion_matrix
###########################################################################################


## Scikit-learnin example 1: array method

y_true1 = [2, 0, 2, 2, 0, 1]
y_pred1 = [0, 0, 2, 2, 0, 2]

C1 = confusion_matrix(y_true1, y_pred1)

print('Example 1:')
print(C1)


## Example 2: list method

y_true2 = ["cat", "ant", "cat", "cat", "ant", "bird"]
y_pred2 = ["ant", "ant", "cat", "cat", "ant", "cat"]

C2 = confusion_matrix(y_true2, y_pred2,labels=["ant", "bird", "cat"])

print('Example 2:')
print(C2)

## Example 3: Binary case

# In the binary case, we can extract four cases: true negative, false positive, false negative, true positives as follows:

y_true3 = [0, 1, 0, 1]
y_pred3 = [1, 1, 1, 0]

tn, fp, fn, tp = confusion_matrix(y_true3, y_pred3).ravel()

print('Example 3:')
print(tn, fp, fn, tp)


# Plotting confusion matrix


columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(y_true1))]]

confm = confusion_matrix(y_true1, y_pred1)
df_cm = DataFrame(confm, index=columns, columns=columns)

ax = sn.heatmap(df_cm, cmap='Oranges', annot=True)

plt.show()
