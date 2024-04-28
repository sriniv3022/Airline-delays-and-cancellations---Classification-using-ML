#!/usr/bin/env python
# coding: utf-8

# ### Airline delays and cancellations - Classification 

# Link for the dataset: https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations?select=train.csv

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


df_data = pd.read_csv('flightsdata.csv')
df_data


# #### Details of the data
# MONTH:				Month, 
# DAY_OF_WEEK:			Day of Week, 
# DEP_DEL15: 			TARGET Binary of a departure delay over 15 minutes (1 is yes), 
# DISTANCE_GROUP:			Distance group to be flown by departing aircraft, 
# DEP_BLOCK:			Departure block, 
# SEGMENT_NUMBER:			The segment that this tail number is on for the day, 
# CONCURRENT_FLIGHTS:		Concurrent flights leaving from the airport in the same departure block, 
# NUMBER_OF_SEATS:		Number of seats on the aircraft, 
# CARRIER_NAME:			Carrier, 
# AIRPORT_FLIGHTS_MONTH:		Avg Airport Flights per Month, 
# AIRLINE_FLIGHTS_MONTH:		Avg Airline Flights per Month, 
# AIRLINE_AIRPORT_FLIGHTS_MONTH:	Avg Flights per month for Airline AND Airport, 
# AVG_MONTHLY_PASS_AIRPORT:	Avg Passengers for the departing airport for the month, 
# AVG_MONTHLY_PASS_AIRLINE:	Avg Passengers for airline for month, 
# FLT_ATTENDANTS_PER_PASS:	Flight attendants per passenger for airline, 
# GROUND_SERV_PER_PASS:		Ground service employees (service desk) per passenger for airline, 
# PLANE_AGE:			Age of departing aircraft, 
# DEPARTING_AIRPORT:		Departing Airport, 
# LATITUDE:			Latitude of departing airport, 
# LONGITUDE:			Longitude of departing airport, 
# PREVIOUS_AIRPORT:		Previous airport that aircraft departed from, 
# PRCP:				Inches of precipitation for day,
# SNOW:				Inches of snowfall for day,
# SNWD:				Inches of snow on ground for day,
# TMAX:				Max temperature for day,
# AWND:				Max wind speed for day

# The dataset has 4542343 rows, I am going to use a sample of 20000 rows for this project.

# ### Data Preprocessing

# In[3]:


df = df_data.sample(n=20000, random_state=123)


# In[4]:


df


# **TARGET variable is the "DEP_DEL15" which is the Binary value of a departure delay over 15 minutes (1 is yes)**

# In[5]:


# Changing the target variable to y
df.rename(columns={'DEP_DEL15': 'y'}, inplace=True)


# In[6]:


df.y.value_counts()


# In[7]:


X = df.drop(['y'], axis=1)
X


# In[8]:


y = df.y
y.head()


# In[9]:


X.dtypes


# In[10]:


X.isna().sum()


# The dataset has no null values.

# In[11]:


y.isna().sum()


# In[12]:


X.shape


# In[13]:


X.describe()


# ### Exploratory Data Analysis

# In[14]:


plt.hist(df.DAY_OF_WEEK, bins=7, edgecolor='black')
plt.xlabel('Days of the week')
plt.ylabel('Count of days')
plt.title('Histogram of days')


# In[15]:


# checking distributions
X.hist()


# In[16]:


delay_0 = df[df['y'] == 0]['NUMBER_OF_SEATS']
delay_1 = df[df['y'] == 1]['NUMBER_OF_SEATS']

# Histograms
plt.hist(delay_0, alpha=0.5, bins = 20, label='Delay 0')
plt.hist(delay_1, alpha=0.5, bins = 20, label='Delay 1')
plt.legend()
plt.grid()


# Flights with seats around 140 had more number of delays

# In[17]:


sns.countplot(data=df, y='y', hue='DAY_OF_WEEK')


# Saturday had less number of delays compared to other days

# In[18]:


sns.countplot(data=df, y='y', hue='PLANE_AGE')


# Planes around the age of 12 had more number of delays.

# In[19]:


xcorr = X.corr()
xcorr


# In[20]:


fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(xcorr, cmap='coolwarm', annot=True, fmt=".1f", ax=ax)
ax.set_title('Correlation Heatmap')
plt.show()


# AVG_MONTHLY_PASS_AIRPORT and AIRPORT_FLIGHTS_MONTH are highly correlated.

# In[21]:


y


# In[22]:


X.info()


# In[23]:


# Removing non-numeric variables
col_drop = ['DEP_TIME_BLK', 'CARRIER_NAME', 'DEPARTING_AIRPORT', 'PREVIOUS_AIRPORT']
X = X.drop(columns=col_drop)


# In[24]:


X.info()


# In[25]:


# split the datasets into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state = 55, test_size= 0.25)


# In[26]:


y_train.value_counts(normalize=True)


# In[27]:


y_test.value_counts(normalize=True)


# ### Logistic Regression

# In[28]:


from sklearn.linear_model import LogisticRegression


# In[29]:


clf = LogisticRegression(max_iter = 10000, C=0.1)


# In[30]:


# Initiating Logistic Regression and fitting the model
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter = 10000, C=0.1)
clf.fit(X_train, y_train)


# In[31]:


# making predictions
pred_tr = clf.predict(X_train)
pred_tr


# In[32]:


pred_ts = clf.predict(X_test)
pred_ts


# In[33]:


# calculating training and testing accuracy scores
(y_train == pred_tr).mean()


# In[34]:


(y_test == pred_ts).mean()


# In[35]:


from sklearn import metrics
# Creating confusion matrix
confusion_tr = metrics.confusion_matrix(y_train, pred_tr)
confusion_tr


# In[36]:


# accuracy for test
confusion_ts = metrics.confusion_matrix(y_test, pred_ts)
confusion_ts


# In[37]:


from sklearn.metrics import recall_score
# Recall(Sensitivity)
recall = recall_score(y_test, pred_ts)
recall


# In[38]:


from sklearn.metrics import precision_score
# Precision
precision = metrics.precision_score(y_test, pred_ts)
precision


# In[39]:


from sklearn.metrics import f1_score
# Calculating F1 score
f1score = metrics.f1_score(y_test, pred_ts)
f1score


# The F1 score of 0 resembles that the logistic regression model is performing poorly. We'll try to improve performance by normalizing the data.

# ### Normalizing the data to check whether it chnages the performance

# In[40]:


from sklearn import preprocessing


# In[41]:


# Using Standard scalar to scale the data
ss = preprocessing.StandardScaler()
X_train_ss = ss.fit_transform(X_train)
X_test_ss = ss.transform(X_test)


# In[42]:


C_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5]


# In[43]:


# Initializing empty lists
train_accuracy = []
test_accuracy = []

for c in C_list:
    # Logistic regression
    lr_model = LogisticRegression(max_iter = 10000,C=c, random_state=134)
    
    # Fit the model
    lr_model.fit(X_train_ss, y_train)
    
    # Predictions
    y_train_pred = lr_model.predict(X_train_ss)
    # For test
    y_test_pred = lr_model.predict(X_test_ss)
    
    # Accuracy
    train_acc = metrics.accuracy_score(y_train, y_train_pred)
    # test
    test_acc = metrics.accuracy_score(y_test, y_test_pred)
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)

# Numpy array results
train_accuracy = np.array(train_accuracy)
test_accuracy = np.array(test_accuracy)

# Output
for i in range(len(C_list)):
    print(f"C: {C_list[i]}, Train Accuracy: {train_accuracy[i]}, Test Accuracy: {test_accuracy[i]}")


# In[44]:


# Creating a validation curve
plt.figure(figsize=(6, 4))
plt.plot(C_list, train_accuracy, 'o-', label='Train Accuracy')
plt.plot(C_list, test_accuracy, 'o-', label='Test Accuracy')
plt.xlabel('C value')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.legend()
plt.show()


# In[45]:


clf = LogisticRegression(max_iter = 10000, C=0.1)
clf.fit(X_train_ss, y_train)


# In[46]:


# making predictions
pred_tr1 = clf.predict(X_train_ss)
pred_tr1


# In[47]:


pred_ts1 = clf.predict(X_test_ss)
pred_ts1


# In[48]:


# Creating confusion matrix
confusion_tr1 = metrics.confusion_matrix(y_train, pred_tr1)
confusion_tr1


# In[49]:


# accuracy for test
confusion_ts1 = metrics.confusion_matrix(y_test, pred_ts1)
confusion_ts1


# In[50]:


# Recall(Sensitivity)
recall = recall_score(y_test, pred_ts1)
recall


# In[51]:


# Precision
precision = metrics.precision_score(y_test, pred_ts1)
precision


# In[52]:


# Calculating F1 score
f1score = metrics.f1_score(y_test, pred_ts1)
f1score


# Scaling/normalizing didn't improve the performance that much.

# ### K Nearest Neighbours

# In[53]:


from sklearn.neighbors import KNeighborsClassifier


# In[54]:


# clf = KNeighborsClassifier(n_neighbors=1)
# clf = KNeighborsClassifier(n_neighbors=3)
# clf = KNeighborsClassifier(n_neighbors=5)
clf = KNeighborsClassifier(n_neighbors=21)


# In[55]:


# Initiating the classifier and fitting the model
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=21)


# In[56]:


knn = clf.fit(X_train, y_train)


# In[57]:


# predictions
pred_knn = clf.predict(X_train)
pred_knn


# In[58]:


# predicting test data
predts_knn = clf.predict(X_test)
predts_knn


# In[59]:


# accuracy(train)
(y_train == pred_knn).mean()


# In[60]:


# accuracy(test)
(y_test == predts_knn).mean()


# In[61]:


# Confusion matrices
knn_conf_tr = metrics.confusion_matrix(y_train, pred_knn)
knn_conf_tr


# In[62]:


knn_conf_ts = metrics.confusion_matrix(y_test, predts_knn)
knn_conf_ts


# In[63]:


# Recall
recall_knn = recall_score(y_test, predts_knn)
recall_knn


# In[64]:


# Precision
precision_knn = metrics.precision_score(y_test, predts_knn)
precision_knn


# In[65]:


# F1 score
f1score_knn = metrics.f1_score(y_test, predts_knn)
f1score_knn


# 1. k = 1, This model has performed better than the Logistic regression but it is still a poor model for our dataset.
# 2. k = 3, didn't perform better than knn with k =1.
# 3. k = 5, performed as same as knn with k = 3.
# 4. k = 21, didn't perform well.

# In[66]:


# K value
K_list = [1,3,5,7,9,15,17,19,21]


# In[67]:


# Initializing empty lists
train_accuracy = []
test_accuracy = []

for k in K_list:
    # KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Fit the model
    knn.fit(X_train, y_train)
    
    # Predictions
    y_train_pred_knn = knn.predict(X_train)
    y_test_pred_knn = knn.predict(X_test)
    
    # Accuracy
    train_acc_knn = metrics.accuracy_score(y_train, y_train_pred_knn)
    # For test
    test_acc_knn = metrics.accuracy_score(y_test, y_test_pred_knn)
    train_accuracy.append(train_acc_knn)
    test_accuracy.append(test_acc_knn)

# Numpy array results
train_accuracy = np.array(train_accuracy)
test_accuracy = np.array(test_accuracy)

# output
for i in range(len(K_list)):
    print(f"K: {K_list[i]}, Train Accuracy: {train_accuracy[i]}, Test Accuracy: {test_accuracy[i]}")


# In[68]:


# Validation curve
plt.figure(figsize=(6, 4))
plt.plot(K_list, train_accuracy, 'o-', label='Train Accuracy')
plt.plot(K_list, test_accuracy, 'o-', label='Test Accuracy')
plt.xlabel('K value')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.legend()
plt.show()


# In[69]:


y.value_counts()


# There is class imbalance in the data which might be causing overfitting in our model which might be affecting the model performance.

# In[70]:


from sklearn.utils import resample

minor_class = df[df['y'] == 1]
major_class = df[df['y'] == 0]

# Undersampling the majority
unsamp_class = resample(major_class, replace=False, n_samples=len(minor_class), random_state=123)

# Combining undersampled majority class to the minor class
unsamp_df = pd.concat([minor_class, unsamp_class])
unsamp_df = unsamp_df.sample(frac=1, random_state=123).reset_index(drop=True)

# Class distribution
print(unsamp_df['y'].value_counts())


# In[71]:


unsamp_df


# This is a balanced data with equal number of 0s and 1s in the target column. We hope that this would improve the model performance.

# ### Decision Tree

# In[72]:


X = unsamp_df.drop(['y'], axis=1)
X


# In[73]:


# Removing non-numeric variables
col_drop = ['DEP_TIME_BLK', 'CARRIER_NAME', 'DEPARTING_AIRPORT', 'PREVIOUS_AIRPORT']
X = X.drop(columns=col_drop)


# In[74]:


y = unsamp_df.y
y.head()


# In[75]:


# split the datasets into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state = 123, test_size= 0.25)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.50, random_state=123)


# In[76]:


# Decision tree classifier
from sklearn.tree import DecisionTreeClassifier


# In[77]:


dt = DecisionTreeClassifier(random_state=123)


# In[78]:


# Fitting the model
dt.fit(X_train, y_train)


# In[79]:


# Predicting training and validation
y_tr_pred = dt.predict(X_train)
y_val_pred = dt.predict(X_val)


# In[80]:


from sklearn.metrics import accuracy_score
# Training accuracy
acc_dt_train = accuracy_score(y_train, y_tr_pred)
print("Training accuracy: {:.2f}".format( acc_dt_train))

# Validation accuracy
acc_dt_val = accuracy_score(y_val, y_val_pred)
print("Validation accuracy: {:.2f}".format( acc_dt_val))


# In[81]:


# Train
from sklearn.metrics import confusion_matrix
conf_dt_tr = confusion_matrix(y_train, y_tr_pred)
print(conf_dt_tr)


# In[82]:


# For validation
conf_dt_val = confusion_matrix(y_val, y_val_pred)
print(conf_dt_val)


# In[83]:


from sklearn.metrics import classification_report
# Accuracy of the validation
print("Accuracy: {:.2f}".format(accuracy_score(y_val, y_val_pred)))

# Precision of the validation 
print("Precision: {:.2f}".format(precision_score(y_val, y_val_pred)))

# Recall of the validation
print("Recall(Sensitivity): {:.2f}".format(recall_score(y_val, y_val_pred)))

# F1-score of the validation
print("F1-score: {:.2f}".format(f1_score(y_val, y_val_pred)))

#Specificity of the validation 
tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
specificity = tn / (tn + fp)
print("Specificity: {:.2f}".format(specificity))

# Classification report
print('\n')
print("Classification Report:")
print(classification_report(y_val, y_val_pred))


# In[84]:


from sklearn.model_selection import GridSearchCV
# Grid search
param_grid = {'min_samples_split': range(1, 26)}

# Using 5-fold cv
grid_search = GridSearchCV(dt, param_grid, cv=5)  
grid_search.fit(X_train, y_train)

samples_split = grid_search.best_params_ 
best_score = grid_search.best_score_

print("Best Min Samples Split: ", samples_split)
print("Best Score: ", best_score)


# In[85]:


dt1 = DecisionTreeClassifier(min_samples_split=25, random_state=123)
dt1.fit(X_train, y_train)

# Prediction
y_val_pred1 = dt1.predict(X_val)


# In[86]:


print("For min_samples_split = 25:")

# confusion matrix
cm1 = confusion_matrix(y_val, y_val_pred1)
print("Confusion Matrix:")
print(cm1)

# Accuracy of the validation set
print("Accuracy: {:.2f}".format(accuracy_score(y_val, y_val_pred1)))

# Precision of the validation set
print("Precision: {:.2f}".format(precision_score(y_val, y_val_pred1)))

# Recall of the validation set
print("Recall(Sensitivity): {:.2f}".format(recall_score(y_val, y_val_pred1)))

# F1-score of the validation set
print("F1-score: {:.2f}".format(f1_score(y_val, y_val_pred1)))

#Specificity of the validation set
tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred1).ravel()
specificity = tn / (tn + fp)
print("Specificity: {:.2f}".format(specificity))

# Classification report
print('\n')
print("Classification Report:")
print(classification_report(y_val, y_val_pred1))


# In[87]:


from sklearn.model_selection import GridSearchCV

# grid search
param_grid = {'max_depth': range(1, 26)}  # Searching for max_depth from 1 to 20

# using cross-validation
grid_search = GridSearchCV(dt, param_grid, cv=5)  #5-fold cross-validation
grid_search.fit(X_train, y_train)  # Fit the grid search to training data

# best parameters
best_max_depth = grid_search.best_params_  #['max_depth']
best_score = grid_search.best_score_

print("Best Max Depth: ", best_max_depth)
print("Best Score: ", best_score)


# In[88]:


# Decision tree
dt2 = DecisionTreeClassifier(max_depth=5, random_state=123)

# Fitting the model
dt2.fit(X_train, y_train)

# Prediction on validation
y_val_pred2 = dt2.predict(X_val)


# In[89]:


# Performance metrics
print("For max_depth = 5")

#confusion matrix
cm2 = confusion_matrix(y_val, y_val_pred2)
print("Confusion Matrix:")
print(cm2)
print('\n')

# Accuracy of the validation
print("Accuracy: {:.2f}".format(accuracy_score(y_val, y_val_pred2)))

# Precision of the validation
print("Precision: {:.2f}".format(precision_score(y_val, y_val_pred2)))

# Recall of the validation
print("Recall(Sensitivity): {:.2f}".format(recall_score(y_val, y_val_pred2)))

# F1-score of the validation
print("F1-score: {:.2f}".format(f1_score(y_val, y_val_pred2)))

#Specificity of the validation
tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred2).ravel()
specificity = tn / (tn + fp)
print("Specificity: {:.2f}".format(specificity))

# Classification report
print('\n')
print("Classification Report:")
print(classification_report(y_val, y_val_pred2))


# In[90]:


# decision tree
dt3 = DecisionTreeClassifier(max_depth=5, min_samples_split=25, random_state=123)

# Fitting the model
dt3.fit(X_train, y_train)

# Prediction on validation 
y_val_pred3 = dt3.predict(X_val)


# In[91]:


# Performance metrics
print("For max_depth = 5; min_samples_split=25")
print("\n")

#confusion matrix
cm3 = confusion_matrix(y_val, y_val_pred3)
print("Confusion Matrix:")
print(cm3)

# Accuracy of the validation set
print("Accuracy: {:.2f}".format(accuracy_score(y_val, y_val_pred3)))

# Precision of the validation set
print("Precision: {:.2f}".format(precision_score(y_val, y_val_pred3)))

# Recall of the validation set
print("Recall(Sensitivity): {:.2f}".format(recall_score(y_val, y_val_pred3)))

# F1-score of the validation set
print("F1-score: {:.2f}".format(f1_score(y_val, y_val_pred3)))

#Specificity of the validation set
tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred3).ravel()
specificity = tn / (tn + fp)
print("Specificity: {:.2f}".format(specificity))

# Classification report
print('\n')
print("Classification Report:")
print(classification_report(y_val, y_val_pred3))


# In[92]:


import matplotlib.pyplot as plt

def plot_variable_importance(model, feature_names):
    importances = model.feature_importances_
    # Sort 
    indices = np.argsort(importances)[::-1]

    # Plot the variable importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), importances[indices])
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in indices])
    plt.xlabel('Variable Importance')
    plt.title('Variable Importance Plot')
    plt.show()


# In[93]:


plot_variable_importance(dt3, feature_names=X.columns[:])


# In[94]:


dt3.feature_importances_


# The Decision Tree Model has performed a lot better than previous models. 

# ### Random Forest Classifier

# In[95]:


from sklearn.ensemble import RandomForestClassifier


# In[96]:


# code
rf = RandomForestClassifier(random_state=123)
rf.fit(X_train, y_train)
# prediction
r_pred_tr = rf.predict(X_train)
r_pred_val = rf.predict(X_val)


# In[97]:


# Training accuracy
acc_rf_train = accuracy_score(y_train, r_pred_tr)
print("Training accuracy of the random forest model with default parameters: {:.2f}".format( acc_rf_train))

# Validation accuracy
acc_rf_val = accuracy_score(y_val, r_pred_val)
print("Validation accuracy of the random forest model with default parameters: {:.2f}".format( acc_rf_val))


# In[98]:


# Confusion matrix
# Training 
conf_rf_tr = confusion_matrix(y_train, r_pred_tr)
print(conf_rf_tr)


# In[99]:


# Validation
conf_rf_val = confusion_matrix(y_val, r_pred_val)
print(conf_rf_val)


# In[100]:


# Performance metrics
print("Random Forest Classifier:")

# Accuracy of the validation
print("Accuracy: {:.2f}".format(accuracy_score(y_val, r_pred_val)))

# Precision of the validation
print("Precision: {:.2f}".format(precision_score(y_val, r_pred_val)))

# Recall of the validation
print("Recall(Sensitivity): {:.2f}".format(recall_score(y_val, r_pred_val)))

# F1-score of the validation
print("F1-score: {:.2f}".format(f1_score(y_val, r_pred_val)))

#Specificity of the validation
tn, fp, fn, tp = confusion_matrix(y_val, r_pred_val).ravel()
specificity = tn / (tn + fp)
print("Specificity: {:.2f}".format(specificity))

# Classification report
print('\n')
print("Classification Report:")
print(classification_report(y_val, r_pred_val))


# In[101]:


# grid search
param_grid = {
    'max_depth': range(1, 26),  
    'min_samples_split': [5, 10, 20, 50], 
    'n_estimators': [1, 2, 5, 10, 15]  
}

# using cross-validation
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs = -1)
grid_search.fit(X_train, y_train) 

# best parameters
best_max_depth = grid_search.best_params_['max_depth']
best_min_samples_split = grid_search.best_params_['min_samples_split']
best_n_estimators = grid_search.best_params_['n_estimators']
best_score = grid_search.best_score_

print("Best Max Depth: ", best_max_depth)
print("Best Min Samples Split: ", best_min_samples_split)
print("Best n_estimators: ", best_n_estimators)
print("Best Score: ", best_score)


# In[102]:


# random forest classifier
rf1 = RandomForestClassifier(min_samples_split=50, max_depth=6, n_estimators=15, random_state=123)

# Fitting the model
rf1.fit(X_train, y_train)

# Prediction on validation
rf_pred_val1 = rf1.predict(X_val)


# In[103]:


print("Hyperparameters of Random Forest Classifier: min_samples_split = 50, max_depth=6, n_estimators=15")
print("\n")

#confusion matrix
cm1_rf = confusion_matrix(y_val, rf_pred_val1)
print("Confusion Matrix:")
print(cm1_rf)
print("\n")

# Accuracy of the validation
print("Accuracy: {:.2f}".format(accuracy_score(y_val, rf_pred_val1)))

# Precision of the validation
print("Precision: {:.2f}".format(precision_score(y_val, rf_pred_val1)))

# Recall of the validation
print("Recall(Sensitivity): {:.2f}".format(recall_score(y_val, rf_pred_val1)))

# F1-score of the validation
print("F1-score: {:.2f}".format(f1_score(y_val, rf_pred_val1)))

#Specificity of the validation
tn, fp, fn, tp = confusion_matrix(y_val, rf_pred_val1).ravel()
specificity = tn / (tn + fp)
print("Specificity: {:.2f}".format(specificity))

# Classification report
print('\n')
print("Classification Report:")
print(classification_report(y_val, rf_pred_val1))


# In[104]:


plot_variable_importance(rf1, feature_names=X.columns[:])


# The Random Forest model has been the best model used so far in this project.

# ### Adaboost

# In[105]:


from sklearn.ensemble import AdaBoostClassifier


# In[106]:


# YOUR CODE
ab = AdaBoostClassifier()

#Fitting the model
ab.fit(X_train, y_train)


# In[107]:


#Accuracy for train and test
ab_pred_train = ab.predict(X_train)
print("Accuracy, Adaboost Classifier for Training: ", metrics.accuracy_score(y_train, ab_pred_train))

ab_pred_val = ab.predict(X_val)
print("Accuracy, Adaboost Classifier for Testing: ", metrics.accuracy_score(y_val, ab_pred_val))


# In[108]:


# confusion matrix
conf_mat = pd.crosstab(index=np.ravel(y_val), columns=ab_pred_val.ravel(), rownames=['Actual'], colnames=['Predicted'])
conf_mat


# In[109]:


# Performance matrix
print("Adaboost Classifier:")

#Classification Report
print("Classification Report:")
print('\n')
ab_met = metrics.classification_report(y_val, ab_pred_val,  output_dict=True)
print(metrics.classification_report(y_val, ab_pred_val))

# performance metrics
print("\nPerformance Metrics:")

# Sensitivity
print("Sensitivity: %f"%(ab_met['1']['recall']))
# Specificity
print("Specificity: %f"%(ab_met['0']['recall']))
# Precision
print("Precision: %f"%(ab_met['1']['precision']))
# Accuracy
print("Accuracy: %f"%(ab_met['accuracy']))
# F1-score
print("F1-score: %f"%(ab_met['1']['f1-score']))


# In[110]:


from sklearn.model_selection import cross_val_score
L_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
acc_valid = []
for lr in L_rate:
    ada_bst = AdaBoostClassifier(learning_rate = lr, random_state = 123)
    ada_bst.fit(X_train, y_train)
    pred_train = ada_bst.predict(X_train)
    score_valid = cross_val_score(ada_bst, X_train, y_train, scoring = "f1", cv = 5)
    acc_valid.append(score_valid.mean())
print("f1 Score")
acc_valid


# In[111]:


plt.figure(figsize=(6,4))
plt.title("The Validation Curve")
plt.plot(acc_valid, 'go-',label = 'validation')
plt.xticks(np.arange(len(L_rate)), L_rate, rotation = 45)
plt.xlabel('LR')
plt.ylabel('f1 Scores')
plt.legend()
plt.show()


# In[112]:


# Computing the estimates
Est_list = [10, 50, 100, 200, 300, 400]
acc_valid = []
for et in Est_list:
    ada_bst = AdaBoostClassifier(n_estimators=et, random_state = 123)
    ada_bst.fit(X_train, y_train)
    pred_train = ada_bst.predict(X_train)
    score_valid= cross_val_score(ada_bst,X_train, y_train, scoring = "f1", cv =5 )
    acc_valid.append(score_valid.mean())
print("f1 score")
acc_valid


# In[113]:


# Plotting the validation curve
plt.figure(figsize=(6,4))
plt.title("The Validation Curve")
plt.plot(acc_valid, 'mo-',label = 'validation')
plt.xticks(np.arange(len(Est_list)), Est_list, rotation = 45)
plt.xlabel('N of Est')
plt.ylabel('f1 Scores')
plt.legend()
plt.show()


# In[114]:


# grid search
L_rate_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
Est_list = [10, 50, 100, 200, 300, 400]
params = {'n_estimators': Est_list,
         'learning_rate': L_rate_list, 'random_state': [123]}
ada_bst = AdaBoostClassifier()
#Grid Search
grid = GridSearchCV(estimator=ada_bst, param_grid=params, cv=5, verbose =1, scoring="f1", n_jobs=-1)


# In[115]:


grid.fit(X_train, y_train)


# In[116]:


grid.best_params_


# In[117]:


grid.best_estimator_


# In[118]:


# Grid Search
ada_bst = AdaBoostClassifier(learning_rate=0.001, n_estimators=10, random_state = 123)
ada_bst.fit(X_train, y_train)


# In[119]:


# Accuracy for Train and Test
pred_train = ada_bst.predict(X_train)
print("Accuracy Scores for Train: ",metrics.accuracy_score(y_train, pred_train))
pred = ada_bst.predict(X_val)
print("Accuracy Scores for Test: ",metrics.accuracy_score( y_val, pred))


# In[120]:


#Calculating the performance metrics
# confusion matrix
conf_mat = pd.crosstab(index=np.ravel(y_val), columns=pred.ravel(), rownames=['Actual'], colnames=['Predicted'])
conf_mat
# confusion matrix
plt.show()
ada_bst_met = metrics.classification_report(y_true = y_val, y_pred = pred,  output_dict=True)
print(metrics.classification_report(y_true = y_val, y_pred = pred))
# Calculating the performance metrics
print("\nPerformance Metrics:")
# Sensitivity
print("Sensitivity %f"%(ada_bst_met['1']['recall']))
# Specificity
print("Specificity %f"%(ada_bst_met['0']['recall']))
# Precision
print("Precision %f"%(ada_bst_met['1']['precision']))
# Accuracy
print("Accuracy %f"%(ada_bst_met['accuracy']))
# F1-score
print("F1-score %f"%(ada_bst_met['1']['f1-score']))


# In[121]:


plot_variable_importance(ada_bst, feature_names=X.columns[:])


# The Adaboost model has performed slightly better than the Random Forest.

# ### Gradient Boosting Machine

# In[122]:


from sklearn.ensemble import GradientBoostingClassifier
grd_bst = GradientBoostingClassifier()
grd_bst.fit(X_train, y_train)
pred_train = grd_bst.predict(X_train)
print("Accuracy score of Gradient Boosting for training data:", metrics.accuracy_score(y_true = y_train, y_pred = pred_train))
pred = grd_bst.predict(X_val)
print("Accuracy score of Gradient Boosting for testing data:", metrics.accuracy_score(y_true = y_val, y_pred = pred))


# In[123]:


# confusion matrix
conf_mat = pd.crosstab(index=np.ravel(y_val), columns=pred.ravel(), rownames=['Actual'], colnames=['Predicted'])
print(conf_mat)


# In[124]:


print("Gradient Boosting Classifier:")

# Classification Report
print('Classification Report:')
print("\n")
grd_bst_met = metrics.classification_report(y_true = y_val, y_pred = pred,  output_dict=True)
print(metrics.classification_report(y_true = y_val, y_pred = pred))

# Calculating the performance metrics
print("\nPerformance Metrics:")
print("\n")

# Sensitivity
print("Sensitivity %f"%(grd_bst_met['1']['recall']))
# Specificity
print("Specificity %f"%(grd_bst_met['0']['recall']))
# Precision
print("Precision %f"%(grd_bst_met['1']['precision']))
# Accuracy
print("Accuracy %f"%(grd_bst_met['accuracy']))
# F1-score
print("F1-score %f"%(grd_bst_met['1']['f1-score']))


# In[125]:


# Calculating the Learning Rate
L_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
acc_valid = []
for lr in L_rate:
    grd_bst = GradientBoostingClassifier(learning_rate=lr, random_state =123)
    grd_bst.fit(X_train, y_train)
    pred_train = grd_bst.predict(X_train)
    score_valid= cross_val_score(grd_bst,X_train, y_train, scoring = "f1", cv =5 )
    acc_valid.append(score_valid.mean())
print("F1 score: ")
acc_valid


# In[126]:


# Plotting the 5-fold for Gradient Boosting classifier
plt.figure(figsize=(6,4))

#Title
plt.title("The Validation Curve of 5-fold GB")
plt.plot(acc_valid, 'co-',label = 'validation')
plt.xticks(np.arange(len(L_rate_list)), L_rate_list, rotation = 45)
plt.xlabel('Learning Rates')
plt.ylabel('F1 Scores')
plt.legend()
plt.show()


# In[127]:


# Calculating estimates
Est_list = [10, 50, 100, 200, 300, 400]
acc_valid = []
for et in Est_list:
    grd_bst = GradientBoostingClassifier(n_estimators=et, random_state =123)
    grd_bst.fit(X_train, y_train)
    pred_train = grd_bst.predict(X_train)
    score_valid= cross_val_score(grd_bst,X_train, y_train, scoring = "f1", cv =5 )
    acc_valid.append(score_valid.mean())
print("F1 score")
acc_valid


# In[128]:


# Plotting the validation curve
plt.figure(figsize=(6,4))
plt.title("Validation Curve")
plt.plot(acc_valid, 'co-',label = 'validation')
plt.xticks(np.arange(len(Est_list)), Est_list, rotation = 45)
plt.xlabel('Number of Estimators')
plt.ylabel('f1 Scores')
plt.legend()
plt.show()


# In[129]:


# Performing grid search
L_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
Est_list = [10, 50, 100, 200, 300, 400]
params = {'n_estimators': Est_list,
         'learning_rate': L_rate, 'random_state': [123]}
grd_bst = GradientBoostingClassifier()
#For f1 score
grid = GridSearchCV(estimator=grd_bst, param_grid=params, cv=5, verbose =1, scoring="f1", n_jobs=-1)


# In[130]:


grid.fit(X_train, y_train)


# In[131]:


grid.best_params_


# In[132]:


grid.best_estimator_


# In[133]:


# grid search
grd_bst = GradientBoostingClassifier(learning_rate=0.01, n_estimators=50, random_state =123)
# grid search  
grd_bst.fit(X_train, y_train)


# In[134]:


# accuracy score
pred_train = grd_bst.predict(X_train)
print("Accuracy Scores for train: ",metrics.accuracy_score(y_true = y_train, y_pred = pred_train))
pred = grd_bst.predict(X_val)
print("Accuracy Scores for test: ",metrics.accuracy_score(y_true = y_val, y_pred = pred))


# In[135]:


# confusion matrix 
conf_mat = pd.crosstab(index=np.ravel(y_val), columns=pred.ravel(), rownames=['Actual'], colnames=['Predicted'])
print(conf_mat)

grd_bst_met = metrics.classification_report(y_true = y_val, y_pred = pred,  output_dict=True)
print(metrics.classification_report(y_true = y_val, y_pred = pred))

# performance metrics
print("\nPerformance Metrics:")
#Sensitivity
print("Sensitivity %f"%(grd_bst_met['1']['recall']))
#Specificity
print("Specificity %f"%(grd_bst_met['0']['recall']))
#Precision
print("Precision %f"%(grd_bst_met['1']['precision']))
#Accuracy
print("Accuracy %f"%(grd_bst_met['accuracy']))
#F1-score
print("F1-score %f"%(grd_bst_met['1']['f1-score']))


# In[136]:


plot_variable_importance(grd_bst, feature_names=X.columns[:])


# The Gradient boosting has performed very similar to the Random Forest. Adaboost is still the best model in this project.

# From the above used classifiers in this project the variable **DEP_BLOCK_HIST** has been the most important variable for prediction.

# ### Results:
# 1. The models Logistic Regression and KNN didn't perform well for the data, even after normalizing the data the performance was not improving. After further analysis we found that it might be due to the class imbalance.
# 2. Then we used a sample data which was balanced.
# 3. We tried 4 more classification models[Decision Trees, Random Forest, Adaboost and Gradient Boosting machine] with different parameters.
# 4. Random forest, Adaboost and Gradient Boosting models performed well. Out of those Adaboost was slightly better than other two models.

# In[ ]:




