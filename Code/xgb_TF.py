
# coding: utf-8

# In[1]:


from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from math import floor, sqrt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier
import ModelEvaluation_
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


X_data=np.load('./dataset/X_matrix.npy')
Y=pd.read_pickle('./dataset/Y.p')
Data_order=Y.index.values.astype(np.int16)
Y_data=Y.PSI.values.astype(np.float32)
Y_data[Y_data <= 0.10] = 0
Y_data[Y_data >= 0.70] = 1
data_selected = (Y_data == 0) + (Y_data == 1)
x_data = X_data[data_selected]
y_data = Y_data[data_selected]
data_order = Data_order[data_selected]
y_data=y_data.astype(np.int8)


# In[3]:


# Remove the location information
x_data=x_data.sum(axis=2)>0


# In[4]:


x_data.shape


# ## Split the data into 80% training and 20% testing

# In[5]:


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=6666)


# In[32]:


model = XGBClassifier(
                    n_estimators=500, learning_rate=0.1,
                    random_state=6666, n_jobs=-1,
                    max_depth=floor(sqrt(x_train.shape[1]))
                    )


# In[33]:


model.fit(x_train, y_train)
y_pred = model.predict(x_test)
probabilities = model.predict_proba(x_test)


# In[34]:


acc = metrics.accuracy_score(y_test, y_pred)
auc = metrics.roc_auc_score(y_test, probabilities[:,1])
print('ACC: {}'.format(acc))
print('AUC: {}'.format(auc))


# ## Find Importance Features

# In[35]:


X=pd.read_pickle('./dataset/X.p')
TF_list=np.sort(np.unique(X[['TF']]))


# In[10]:


top_k_imp=50
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
imp_TF_rank=TF_list[indices]

# Print the feature ranking
print("Feature ranking:")

for f in range(top_k_imp):
    print("%d. %s (%f)" % (f + 1, imp_TF_rank[f],  importances[indices[f]]))


# In[11]:


# Plot the feature importances of the Xgboost forest
fig = plt.figure(figsize=(20,10))
plt.title("Top 50 Feature importances")
plt.bar(range(top_k_imp), importances[indices][:top_k_imp], color='deepskyblue', align="center")
plt.xticks(range(top_k_imp), imp_TF_rank[:top_k_imp],rotation=45,horizontalalignment="right")
plt.xlim([-1, top_k_imp])
plt.xlabel('TF(Transcription Factor)')
plt.ylabel('Importance Score')

plt.show()


# ## Evaluate the Model with Confusion Matrix, F1-Score, ROC Curve, AUC...

# ### Confusion Matrix
# 
# |                | Predicted Positive | Predicted Negative |
# |:--------------:|:------------------:|:------------------:|
# |Actual Positive | TP (True Positive) | FN (False Negative)|
# |Actual Negative | FP (False Positive)| TN (True Negative) |

# In[12]:


cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
classes = np.asarray(['Spliced', 'Retained'])
ModelEvaluation_.plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True)


# ### F1-Score
# $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$
# 
# $Precision = \frac{TP}{TP + FP}$
# 
# $Recall = \frac{TP}{TP + TN}$
# 
# $F1-score =2 \times {\frac{2}{\frac{1}{Recall} + \frac{1}{Precision}}} = 2 \times{\frac{Precision \times Recall}{Precision + Recall}}$

# In[13]:


from sklearn.metrics import precision_recall_fscore_support
precision,recall,f1_score,support=precision_recall_fscore_support(y_test, y_pred, average='macro')
print('Precision: {}'.format(precision))
print('Recall: {}'.format(recall))
print('F1 Score: {}'.format(f1_score))


# ### ROC Curve and AUC

# In[14]:


from sklearn.metrics import roc_curve, auc
from itertools import cycle


# In[15]:


def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    
    Arguments:
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    encoding = np.eye(2)[x]
    return encoding


# In[16]:


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(one_hot_encode(y_test)[:, i], probabilities[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[17]:


plt.figure()
lw = 2
#plt.plot(fpr[0], tpr[0], color='darkorange',
#         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
colors = cycle(['aqua', 'darkorange'])
for i, color in zip(range(2), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# ## 5-fold Cross-Validation

# In[23]:


data_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=6666)


# In[24]:


def model_fit(x_train, x_test, y_train, y_test):
    """
    Fit model and return y_pred, acc, auc
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    probabilities = model.predict_proba(x_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, probabilities[:,1])
    return y_pred, acc, auc


# In[60]:


cv_score = defaultdict()
cnf_matrix = []
importances = []
first_fold = True
fold = 1

for train_index, test_index in data_split.split(x_data, y_data, data_order):
    print('The %d th fold: ' % fold)
    X_train, X_test = x_data[train_index], x_data[test_index]
    Y_train, Y_test = y_data[train_index], y_data[test_index]
    model = XGBClassifier(
                    n_estimators=500, learning_rate=0.1,
                    random_state=6666, n_jobs=-1,
                    max_depth=floor(sqrt(x_train.shape[1]))
                    )
    if first_fold:
        cv_score['cv'] = {'acc': [], 'auc': []}
                    
    y_pred, acc, auc = model_fit(X_train, X_test, Y_train, Y_test)
    cv_score['cv']['acc'].append(acc)
    cv_score['cv']['auc'].append(auc)
    print('Accuracy: %.2f%%' % (acc * 100.0))
    print('AUC: %.3f' % auc)
    first_fold = False
    
    cnf = metrics.confusion_matrix(Y_test, y_pred)
    cnf_matrix.append(cnf)
    print('Confusion Matrix: \n',cnf)
    
    imp = model.feature_importances_
    importances.append(imp.tolist())
    top_10_indices = np.argsort(imp)[::-1][:10]
    top_10_TF = TF_list[top_10_indices]
    print('Top 10 Importance Features: \n')
    print(top_10_TF)
    fold+=1
    print()


# In[61]:


avg_acc = ModelEvaluation_.cross_validation_scores(cv_score['cv']['acc'], 'percentage')
avg_auc = ModelEvaluation_.cross_validation_scores(cv_score['cv']['auc'], 'decimal')

print('Accuracy: {}'.format(avg_acc))
print('AUC: {}'.format(avg_auc))


# In[62]:


cnf_matrix = sum(cnf_matrix)
np.set_printoptions(precision=2)
classes = np.asarray(['Spliced', 'Retained'])
ModelEvaluation_.plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True)


# In[63]:


top_k_imp=50
importances=np.array(importances)
avg_importances = np.mean(importances,axis=0)
indices = np.argsort(avg_importances)[::-1]
imp_TF_rank=TF_list[indices]

# Print the feature ranking
print("Feature ranking:")

for f in range(top_k_imp):
    print("%d. %s (%f)" % (f + 1, imp_TF_rank[f],  avg_importances[indices[f]]))


# In[64]:


# Plot the feature importances of the Xgboost forest
fig = plt.figure(figsize=(20,10))
plt.title("Top 50 Feature importances")
plt.bar(range(top_k_imp), avg_importances[indices][:top_k_imp], color='deepskyblue', align="center")
plt.xticks(range(top_k_imp), imp_TF_rank[:top_k_imp],rotation=45,horizontalalignment="right")
plt.xlim([-1, top_k_imp])
plt.xlabel('TF(Transcription Factor)')
plt.ylabel('Importance Score')
plt.show()

