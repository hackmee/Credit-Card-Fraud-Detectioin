# Credit-Card-Fraud-Detectioin
import numpy as np # linear algebra
import pandas as pd # data  processing, CSV file
import sys # system-spesific parameters and functions
from sklearn.preprocessing import StandardScaler # Scale the feautres
import imblearn # Handling Imballance data
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler # Resampling Technique
# from imblearn.over_sampling import RandomOverSampler # Resampling Technique
# from imblearn.over_sampling import SMOTE  # Resampling Technique
# from imblearn.combine import SMOTETomek  # Resampling Technique
# from imblearn.under_sampling import TomekLinks # Resampling Technique
# from imblearn.over_sampling import ADASYN  # Resampling Technique


# Classifiers and Modeling Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier



# Features Importances - Selection Libraries
from sklearn.ensemble import ExtraTreesClassifier # This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
from sklearn.feature_selection import SelectFromModel # Meta-transformer for selecting features based on importance weights.


# Performance Metrics and Visualisations
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time

import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns # visualise random distributions. It uses matplotlb




# read the data using the pandas library
dataset = pd.read_csv('creditcard.csv', header = 0, comment='\t', sep = ",")



### Data Exploration ###



# read the first five rows
dataset.head() 
#check out the dimension of the dataset
dataset.shape 

# Obervations:
    # Interesting info (memory_usage, null_counts = 0)
    # We see that we have only numerical values so no need to transform categorical ones into dummy variables and also non-null values

# Print the full summary and the columns 
dataset.info() 
dataset.columns




## Desrriptive Statistics


# Summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
dataset.describe()
# As most of the columns V1, V2,... V28 are transformed using PCA so neither features make much sense and nor will the descriptive statistics so we will leave them and consider only Time and Amount which makes sense. 
dataset[['Time', 'Amount']].describe()

# Observations:
    # Mean transaction is somewhere is 88 and standard deviation is around 250.
    # The median is 22 which is very less as compared to mean which signifies that there are outliers or our data is highly positive skewed which is effecting the amount and thus the mean. 
    # The maximum transaction that was done is of 25,691 and minimum is 0.


# Check the percentages of fraudulent and non-fraudulent transactions
majority, minority = np.bincount(dataset['Class'])
total = majority + minority



print('Examples:\n    Total: {}\n    Minority: {} ({:.2f}% of total)\n'.format(
    total, minority, 100 * minority / total))
print(f'Percent of Non-Fraudulent Transactions(Majority) = {round(dataset["Class"].value_counts()[0]/len(dataset) * 100,2)}%') # 
print(f'Percent of Fraudulent Transactions(Minority) = {round(dataset["Class"].value_counts()[1]/len(dataset) * 100,2)}%')


# Observations:
    # Only 492 (or 0.17%) of transaction are fraudulent. That means the data is highly unbalanced with respect with target variable Class.
    # Most of the transactions are legitimate. In case we use this data to predtict the frauds, our algorithms will overfit. There will be a bias towards the majority class and the accuracy of the models will be misleading. 
    # So, later on, we will balance the data to make the algorithms to produce reliable results.



# Feature Correlation with Response to the label(Class)
corr = dataset.corrwith(dataset['Class']).reset_index()
corr.columns = ['Index','Correlations']
corr = corr.set_index('Index')
corr = corr.sort_values(by=['Correlations'], ascending = True)
plt.figure(figsize=(9, 12))
fig = sns.heatmap(corr, annot=True, fmt="g", cmap='Set3', linewidths=0.3, linecolor='black')
plt.title("Feature Correlation with Class", fontsize=18)
plt.show()


# Observations:
    # V17, V14, V12 and V10 are negatively correlated. Notice how the lower these values are, the more likely the end result will be a fraud transaction.
    # V2, V4, V11, and V19 are positively correlated. Notice how the higher these values are, the more likely the end result will be a fraud transaction.
    # For some of the features we can observe a good selectivity in terms of distribution for the two values of Class: V4, V11 have clearly separated distributions for Class values 0 and 1,
    # V12, V14, V18 are partially separated, V1, V2, V3, V10 have a quite distinct profile, whilst V20-V28 have similar profiles for the two values of Class and thus not very useful in differentiation of both the classes.
    # In general, with just few exceptions (Time and Amount), the features distribution for legitimate transactions (values of Class = 0) is centered around 0, sometime with a long queue at one of the extremities. 
    # In the same time, the fraudulent transactions (values of Class = 1) have a skewed (asymmetric) distributio.



# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)
scaled_dataset = dataset.copy()

std_scaler = StandardScaler()

scaled_dataset ['scaled_amount'] = std_scaler.fit_transform(scaled_dataset ['Amount'].values.reshape(-1,1))
scaled_dataset ['scaled_time'] = std_scaler.fit_transform(scaled_dataset ['Time'].values.reshape(-1,1))

scaled_dataset .drop(['Time','Amount'], axis=1, inplace=True)
scaled_amount = scaled_dataset ['scaled_amount']
scaled_time = scaled_dataset ['scaled_time']

scaled_dataset .drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
scaled_dataset .insert(0, 'scaled_amount', scaled_amount)
scaled_dataset .insert(1, 'scaled_time', scaled_time)
print(scaled_dataset.describe())



##############################################################################


##############################################################################



### Data Manipulation ###



# Before proceeding with the Random UnderSampling technique we have to separate the orginal dataframe.
# We do this because we want to test our models on the original testing set and not on the testing set created by the Random UnderSampling technique.
# Also, the resampling technique should be done only on the training set. 



# Data Split for training 80:20
X = scaled_dataset.drop(['Class'], axis=1) # Features
Y = scaled_dataset['Class'] # Labels
# The tes_size is being chosen by general rule
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Check the shape
print(X_train.shape, X_test.shape)






# Resampling Technique - UNDERSAMPLING - Balance the data - Handling imbalanced data Process


# We need ratio = 1 between the two classes
undersample = RandomUnderSampler(sampling_strategy=1, random_state=42) 
X_trainundersam, y_trainundersam = undersample.fit_resample(X_train, y_train)




# Returning to new training set # Concat. # Concatenate pandas objects along a particular axis with optional set logic along the other axes. Can also add a layer of hierarchical indexing on the concatenation axis, which may be useful if the labels are the same (or overlapping) on the passed axis number.
undersamdataset = pd.concat([X_trainundersam, y_trainundersam.reindex(X_trainundersam.index)], axis=1)


# equally distributed
print('Distribution of the Classes in the Undersampling subsample dataset')
print(undersamdataset['Class'].value_counts()/len(undersamdataset))

# Check the difference
print(undersamdataset) 
print(dataset) 


# Separate undersampled data into X and y sets - split features and labels 
X_trainnew = undersamdataset.drop(['Class'], axis=1)  # Features
Y_trainnew = undersamdataset["Class"] # Mono ta lables




"""
# Resampling Technique - OVERSAMPLING - Balance the data - Handling imbalanced data Process
ros = RandomOverSampler(sampling_strategy=1, random_state=42)
X_trainoversam, y_trainoversam = ros.fit_resample(X_train, y_train)
# Returning to new training set # Concat. # Concatenate pandas objects along a particular axis with optional set logic along the other axes. Can also add a layer of hierarchical indexing on the concatenation axis, which may be useful if the labels are the same (or overlapping) on the passed axis number.
oversamdataset = pd.concat([X_trainoversam, y_trainoversam.reindex(X_trainoversam.index)], axis=1)
# check the distribution
print('Distribution of the Classes in the oversampling subsample dataset')
print(oversamdataset['Class'].value_counts()/len(oversamdataset))
# check the difference
print(oversamdataset)
print(dataset)
# Resampling Technique - SMOTE - Balance the data - Handling imbalanced data Process
sm = SMOTE(random_state=42)
X_trainsmote, y_trainsmote = sm.fit_resample(X_train, y_train)
# check the distribution
print('After OverSampling, the shape of train_X: {}'.format(X_trainsmote.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_trainsmote.shape)) 
print("After OverSampling, counts of label '1': {}".format(sum(y_trainsmote == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_trainsmote == 0))) 
# Returning to new training set # Concat. # Concatenate pandas objects along a particular axis with optional set logic along the other axes. Can also add a layer of hierarchical indexing on the concatenation axis, which may be useful if the labels are the same (or overlapping) on the passed axis number.
smote = pd.concat([pd.DataFrame(X_trainsmote), pd.DataFrame(y_trainsmote)], axis=1)
# equally distributed
print('Distribution of the Classes in the SMOTE subsample dataset')
print(smote['Class'].value_counts()/len(smote))
# check the difference
print(smote)
print(dataset)
# Separate SMOTE data into X and y sets - split features and labels 
X_trainnew = smote.drop(['Class'], axis=1)  # Features
print(X_trainnew)
Y_trainnew = smote["Class"] # Mono ta lables
print(Y_trainnew)
# Resampling Technique - SMOTETomek - Balance the data - Handling imbalanced data Process
smtomek = SMOTETomek()
X_smtomek, y_smtomek = smtomek.fit_sample(X_train, y_train)
# Returning to new training set # Concat. # Concatenate pandas objects along a particular axis with optional set logic along the other axes. Can also add a layer of hierarchical indexing on the concatenation axis, which may be useful if the labels are the same (or overlapping) on the passed axis number.
smotetomek = pd.concat([pd.DataFrame(X_smtomek), pd.DataFrame(y_smtomek)], axis=1)
#check the distribution
print('Distribution of the Classes in the SMOTETomek subsample dataset')
print(smotetomek['Class'].value_counts()/len(smotetomek))
# check the difference
print(smotetomek)
print(dataset)
# Separate undersampled data into X and y sets - lit labels and features - Getting the features and labels(train and labesl) - Upodhlwnw ta features kai labels opou me auta tha ekpaideusw to modelo mou
X_trainnew = smotetomek.drop(['Class'], axis=1)  # Features
print(X_trainnew)
Y_trainnew = smotetomek["Class"] # Mono ta lables
print(Y_trainnew)
# Resampling Technique - Balance the data - Handling imbalanced data Process
# TomekLinks undersampling. Exist if the two samples are the nearest neighbors of each other. 
# Only remove samples form the majority class
tl = TomekLinks()
X_tl, y_tl  = tl.fit_sample(X_train, y_train)
# Returning to new training set # Concat. # Concatenate pandas objects along a particular axis with optional set logic along the other axes. Can also add a layer of hierarchical indexing on the concatenation axis, which may be useful if the labels are the same (or overlapping) on the passed axis number.
tldataset = pd.concat([X_tl, y_tl.reindex(X_tl.index)], axis=1)
# Check the distribution
print('Distribution of the Classes in the TomekLinks subsample dataset')
print(tldataset['Class'].value_counts()/len(tldataset))
# Check the difference
print(tldataset) 
print(dataset) 
# Separate undersampled data into X and y sets - split features and labels 
X_trainnew = tldataset.drop(['Class'], axis=1)  # Features
Y_trainnew = tldataset["Class"] # Mono ta lables
# Resampling Technique - ADASYN - Balance the data - Handling imbalanced data Process
ada = ADASYN(sampling_strategy=1, random_state=42)
X_trainadasyn, y_trainadasyn = ada.fit_resample(X_train, y_train)
# Returning to new training set # Concat. # Concatenate pandas objects along a particular axis with optional set logic along the other axes. Can also add a layer of hierarchical indexing on the concatenation axis, which may be useful if the labels are the same (or overlapping) on the passed axis number.
adasyndataset = pd.concat([X_trainadasyn, y_trainadasyn.reindex(X_trainadasyn.index)], axis=1)
# check the distribution
print('Distribution of the Classes in the ADASYN subsample dataset')
print(adasyndataset['Class'].value_counts()/len(adasyndataset))
# Check the difference
print(adasyndataset) 
print(dataset) 
# Separate undersampled data into X and y sets - split features and labels 
X_trainnew = adasyndataset.drop(['Class'], axis=1)  # Features
Y_trainnew = adasyndataset["Class"] # Mono ta lables
"""

##############################################################################


##############################################################################



### Feature Selection ###



# Selecting features with the ExtraTressClassifier and SelectFromModel.
# Note: ExtraTreesClassifier tends to be biased. But This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

etcmodel = ExtraTreesClassifier(n_estimators=100, criterion = 'entropy', random_state=42)
etcmodel.fit(X_trainnew, Y_trainnew)
feat_labels = X_trainnew.columns.values
#print(feat_labels)
feat_import = etcmodel.feature_importances_
#print(feat_import)

importances = etcmodel.feature_importances_
std = np.std([tree.feature_importances_ for tree in etcmodel.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]


# Print the feature ranking
print("Feature ranking:")
for f in range(X_trainnew.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# Plot the impurity-based feature importances of the ExtraTreesClassifier
plt.figure()
plt.xlabel("Features")
plt.ylabel("Features Importance")
plt.title("Feature importances")
plt.bar(range(X_trainnew.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X_trainnew.shape[1]), indices)
plt.xlim([-1, X_trainnew.shape[1]])
plt.show()



# Select the most important Values
# We will use SelectFromModel, using a threshold to extract the most important features
# Setting the threshold for which variables to keep based on their variance



sfm = SelectFromModel(etcmodel, threshold=0.03, prefit=True)
print('Number of features before selection: {}'.format(X_trainnew.shape[1]))
# Number of features before selection: 30

# Throwing away all the variables which fall below the threshold level␣,→specified above
n_features = sfm.transform(X_trainnew).shape[1]
print('Number of features after selection: {}'.format(n_features))
# Number of features after selection: 10

#Create a kist
selected_features = list(feat_labels[sfm.get_support()])




# split features and labels adding only the selected features
X_trainfinal = undersamdataset[selected_features]
X_testfinal = X_test[selected_features]


#check the difference
print(X_trainfinal)
print(X_trainnew)

# The training and testing should have the same features
print(X_trainfinal.columns)
print(X_testfinal.columns)
#check the difference
print(X_testfinal)



##############################################################################


##############################################################################

# Train datasets: X_trainfinal, Y_trainnew
# Test datasets:  X_testfinal, y_test



### Data Modeling ###




# Build the Logistic Regression model

# Build and calculate the classifier's process
start = time.time()
clfLR = LogisticRegression()
clfLR.fit(X_trainfinal, Y_trainnew)
y_predLR = clfLR.predict(X_testfinal)
end = time.time()
print("This is the time of LR = ", end - start)


# Performance Metrics
print('Logistic Regression Metrics-Score:')
#Confusion Matrix
confusion_matrix1 = confusion_matrix(y_test, y_predLR)
print("	", "pred no", "pred yes")
print("actual no", confusion_matrix1[0])  
print("actual yes", confusion_matrix1[1])

# accuracy: (tp + tn) / (p + n)
accuracy_LR = accuracy_score(y_test, y_predLR)
print('Accuracy: %f' % accuracy_LR)

# precision tp / (tp + fp)
precision_LR = precision_score(y_test, y_predLR)
print('Precision: %f' % precision_LR)

# recall: tp / (tp + fn)
recall_LR = recall_score(y_test, y_predLR)
print('Recall: %f' % recall_LR)

# f1: 2 tp / (2 tp + fp + fn)
f1_LR = f1_score(y_test, y_predLR)
print('F1 score: %f' % f1_LR)

#AUC_ROC
auc_LR = roc_auc_score(y_test, y_predLR)
print('ROC_AUC score: ', roc_auc_score(y_test, y_predLR))

# Classification report
labels = ['No Fraud', 'Fraud']
print(classification_report(y_test, y_predLR, target_names=labels))





# Built Naive Bayes model

# Build and calculate the classifier's process
start = time.time()
clfNB = GaussianNB()
clfNB.fit(X_trainfinal, Y_trainnew) 
y_predNB = clfNB.predict(X_testfinal)
end = time.time()
print("This is the time of NB = ", end - start)


# Performance Metrics
print('Naive Bayes Metrics-Score:')
#Confusion Matrix
confusion_matrix2 = confusion_matrix(y_test, y_predNB)
print("	", "pred no", "pred yes")
print("actual no", confusion_matrix2[0])  
print("actual yes", confusion_matrix2[1])

# accuracy: (tp + tn) / (p + n)
accuracy_NB = accuracy_score(y_test, y_predNB)
print('Accuracy: %f' % accuracy_NB)

# precision tp / (tp + fp)
precision_NB = precision_score(y_test, y_predNB)
print('Precision: %f' % precision_NB)

# recall: tp / (tp + fn)
recall_NB = recall_score(y_test, y_predNB)
print('Recall: %f' % recall_NB)

# f1: 2 tp / (2 tp + fp + fn)
f1_NB = f1_score(y_test, y_predNB)
print('F1 score: %f' % f1_NB)

#AUC_ROC
auc_NB = roc_auc_score(y_test, y_predNB)
print('ROC_AUC score: ', roc_auc_score(y_test, y_predNB))

# Classification report
labels = ['No Fraud', 'Fraud']
print(classification_report(y_test, y_predNB, target_names=labels))





# Build Random Forest model

# Build and calculate the classifier's process
start = time.time()
clfRF = RandomForestClassifier(n_estimators=100, random_state=42)
clfRF.fit(X_trainfinal, Y_trainnew) 
y_predRF = clfRF.predict(X_testfinal)
end = time.time()
print("This is the time of RF = ", end - start)


# Performance Metrics
print('Random Forests Metrics-Score:')
#Confusion Matrix
confusion_matrix3 = confusion_matrix(y_test, y_predRF)
print("	", "pred no", "pred yes")
print("actual no", confusion_matrix3[0])  
print("actual yes", confusion_matrix3[1])

# accuracy: (tp + tn) / (p + n)
accuracy_RF = accuracy_score(y_test, y_predRF)
print('Accuracy: %f' % accuracy_RF)

# precision tp / (tp + fp)
precision_RF = precision_score(y_test, y_predRF)
print('Precision: %f' % precision_RF)

# recall: tp / (tp + fn)
recall_RF = recall_score(y_test, y_predRF)
print('Recall: %f' % recall_RF)

# f1: 2 tp / (2 tp + fp + fn)
f1_RF = f1_score(y_test, y_predRF)
print('F1 score: %f' % f1_RF)

#AUC_ROC
auc_RF = roc_auc_score(y_test, y_predRF)
print('ROC_AUC score: ', roc_auc_score(y_test, y_predRF))

# Classification report
labels = ['No Fraud', 'Fraud']
print(classification_report(y_test, y_predRF, target_names=labels))





# Build the Support Vector Machines model

# Build and calculate the classifier's process
start = time.time()
clfSVM = svm.SVC()
clfSVM.fit(X_trainfinal, Y_trainnew)  
y_predSVM = clfSVM.predict(X_testfinal)
end = time.time()
print("This is the time of SVM = ", end - start)

# Performance Metrics
print('Support Vector Machines Metrics-Score:')
#Confusion Matrix
confusion_matrix4 = confusion_matrix(y_test, y_predSVM)
print("	", "pred no", "pred yes")
print("actual no", confusion_matrix4[0])  
print("actual yes", confusion_matrix4[1])

# accuracy: (tp + tn) / (p + n)
accuracy_SVM = accuracy_score(y_test, y_predSVM)
print('Accuracy: %f' % accuracy_SVM)

# precision tp / (tp + fp)
precision_SVM = precision_score(y_test, y_predSVM)
print('Precision: %f' % precision_SVM)

#sys.exit()
# recall: tp / (tp + fn)
recall_SVM = recall_score(y_test, y_predSVM)
print('Recall: %f' % recall_SVM)

# f1: 2 tp / (2 tp + fp + fn)
f1_SVM = f1_score(y_test, y_predSVM)
print('F1 score: %f' % f1_SVM)

#AUC_ROC
auc_SVM = roc_auc_score(y_test, y_predSVM)
print('ROC_AUC score: ', roc_auc_score(y_test, y_predSVM))

# Classification report
labels = ['No Fraud', 'Fraud']
print(classification_report(y_test, y_predSVM, target_names=labels))






# All ROC_AUC scores

print('Logistic Regression ROC_AUC Score: ', roc_auc_score(y_test, y_predLR))
print('Random Forests ROC_AUC Score: ', roc_auc_score(y_test, y_predRF))
print('Naive Bayes ROC_AUC Score : ', roc_auc_score(y_test, y_predNB))
print('Support Vector Machines ROC_AUC Score: ', roc_auc_score(y_test, y_predSVM))

log_fpr, log_tpr, log_thresold = roc_curve(y_test, y_predLR)
rf_fpr, rf_tpr, rf_threshold = roc_curve(y_test, y_predRF)
nb_fpr, nb_tpr, nb_threshold = roc_curve(y_test, y_predNB)
svm_fpr, svm_tpr, svm_threshold = roc_curve(y_test, y_predSVM)
#ab_fpr, ab_tpr, ab_threshold = roc_curve(y_test, y_predAB)

def graph_roc_curve_multiple(log_fpr, log_tpr, rf_fpr, rf_tpr, nb_fpr, nb_tpr, svm_fpr, svm_tpr):
    plt.figure(figsize=(16,8))
    plt.title('ROC Curve \n Random Forests have the highest score', fontsize=18)
    plt.plot(log_fpr, log_tpr, label='Logistic Regression Score: {:.4f}'.format(roc_auc_score(y_test, y_predLR)))
    plt.plot(rf_fpr, rf_tpr, label='Random Forests Score: {:.4f}'.format(roc_auc_score(y_test, y_predRF)))
    plt.plot(nb_fpr, nb_tpr, label='Naive Bayes Score: {:.4f}'.format(roc_auc_score(y_test, y_predNB)))
    plt.plot(svm_fpr, svm_tpr, label='Support Vector Machines Score: {:.4f}'.format(roc_auc_score(y_test, y_predSVM)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()

graph_roc_curve_multiple(log_fpr, log_tpr, rf_fpr, rf_tpr, nb_fpr, nb_tpr, svm_fpr, svm_tpr)
plt.show()
