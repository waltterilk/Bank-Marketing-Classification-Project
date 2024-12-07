# Import necessary libraries
import os
import requests
from urllib.parse import urlparse

# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning models and utilities
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

# Model evaluation and selection
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, recall_score, roc_curve
)

# Visualization settings
%matplotlib inline
plt.rcParams["figure.figsize"] = (12, 7)

# Suppress warnings for clean output
import warnings
warnings.filterwarnings('ignore')
download_urls = [
    "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
]
downloads_folder = 'downloads'

def download_file(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    else:
        print(f"Failed to download {url}")
        return False

def extract_all_zips(folder_path):
    zip_files = [f for f in os.listdir(folder_path) if f.endswith('.zip')]
    for zip_file in zip_files:
        zip_path = os.path.join(folder_path, zip_file)
        extract_folder = os.path.splitext(zip_path)[0]
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        os.remove(zip_path)
        print(f"Extracted zip file: {zip_path}")
        extract_all_zips(extract_folder)

def download_extract():
    downloads_folder = 'downloads'
    if not os.path.exists(downloads_folder):
        os.makedirs(downloads_folder)

    for url in download_urls:
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)

        file_path = os.path.join(downloads_folder, filename)
        if download_file(url, file_path):
            print(f"Downloaded file: {file_path}")

            if file_path.endswith('.zip'):
                extract_folder = os.path.splitext(file_path)[0]
                with ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_folder)
                os.remove(file_path)
                print(f"Extracted zip file: {file_path}")
                extract_all_zips(extract_folder)
                
if __name__ == "__main__":
    download_extract()


downloads_folder = 'downloads'
extracted_folder = os.path.join(downloads_folder, 'bank+marketing', 'bank')
csv_file_path = os.path.join(extracted_folder, 'bank-full.csv')

df = pd.read_csv(csv_file_path, delimiter=';')
df

# Explore the data:
# 1. Display the first few rows
# 2. Get data type information
# 3. Get summary statistics
df.head()
df.info()
df.describe()

# Split data into training and testing sets
X = df.drop(columns=['y'])
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values (e.g., replace missing values with the mean of the column)
missing_values = df.isnull().sum()

print("Missing values:")
print(missing_values)

# Encoding categorical features (e.g., one-hot encoding)
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

print("Categorical columns:", categorical_columns)

df_encoded = pd.get_dummies(df, columns=categorical_columns)
df_encoded

# Data Scaling (normalize numerical features)
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

scaler = MinMaxScaler()

df_scaled = df_encoded.copy()
df_scaled[numerical_columns] = scaler.fit_transform(df_scaled[numerical_columns])
df_scaled

# Data Visualization (e.g., histograms, box plots, and correlation matrices)
plt.figure(figsize=(12, 8))
sns.heatmap(df_scaled[numerical_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Histogram of age
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', bins=20, kde=True)
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Bar plot of job
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='job')
plt.title('Bar Plot of Job')
plt.xlabel('Job')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Bar plot of education level
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='education')
plt.title('Bar Plot of Education Level')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# ### **Step 2: Training Three Classification Models**
# Step 2: Training Three Classification Models
# Preprocessing for data
# replacing yes and no from deposit column by 1 and 0 to convert categorical feature to numerical feature
df['y'].replace(to_replace='yes', value=1, inplace=True)
df['y'].replace(to_replace='no',  value=0, inplace=True)
df['y'].head()

# replacing yes and no from loan column by 1 and 0 to convert categorical feature to numerical feature
df['loan'].replace(to_replace='no', value=1, inplace=True)
df['loan'].replace(to_replace='yes',  value=0, inplace=True)
df.head()

# replacing yes and no from default column by 1 and 0 to convert categorical feature to numerical feature
df['default'].replace(to_replace='no', value=1, inplace=True)
df['default'].replace(to_replace='yes',  value=0, inplace=True)
df.head()

df["loan"].value_counts()

# replacing yes and no from housing column by 1 and 0 to convert categorical feature to numerical feature
df['housing'].replace(to_replace='no', value=1, inplace=True)
df['housing'].replace(to_replace='yes',  value=0, inplace=True)
df.head()

# one hot encoding for marital feature to convert categorical feature to numerical feature
# dropping original column
# dropping one of the martial columns

one_hot = pd.get_dummies(df['marital'])
df = df.drop('marital',axis = 1)
df = df.join(one_hot)

# one hot encoding for education feature to convert categorical feature to numerical feature
# dropping original column
# dropping one of the resultant columns

one_hot = pd.get_dummies(df['education'])
df = df.drop('education',axis = 1)
df = df.join(one_hot)
df = df.drop('unknown',axis = 1)
df.head()

df = df.drop('divorced',axis = 1)
df.head()

one_hot = pd.get_dummies(df['job'])
df = df.drop('job',axis = 1)
df = df.join(one_hot)
df = df.drop('unknown',axis = 1)
df.head()

# one hot encoding for contact feature to convert categorical feature to numerical feature
# dropping original column
# dropping one of the contact columns

one_hot = pd.get_dummies(df['contact'])
df = df.drop('contact',axis = 1)
df = df.join(one_hot)
df = df.drop('unknown',axis = 1)
df.head()

# one hot encoding for month feature to convert categorical feature to numerical feature
# dropping original column
# dropping one of the month columns

one_hot = pd.get_dummies(df['month'])
df = df.drop('month',axis = 1)
df = df.join(one_hot)
df = df.drop('dec',axis = 1)
df.head()

# one hot encoding for poutcome feature to convert categorical feature to numerical feature
# dropping original column
# dropping one of the resultant columns

one_hot = pd.get_dummies(df['poutcome'])
df = df.drop('poutcome',axis = 1)
df = df.join(one_hot)
df = df.drop('other',axis = 1)
df.head()

print(df.info())
tempDF=df['y']
df=df.drop('y',axis=1)
df['y']=tempDF
df.head()

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df.drop('y',axis=1))
scaled_features=scaler.transform(df.drop('y',axis=1))
df_feat=pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()

df_feat['y']=tempDF

### Correlation with Class variable 'y' deposit
CorrBank=df_feat.drop("y", axis=1).apply(lambda x: x.corr(df_feat.y))

# Arranging in descending order
Corr2=CorrBank.sort_values(ascending=False)

Corr2.plot.bar()
plt.xlabel("Features", fontsize=15)
plt.ylabel("Correlation", fontsize=15)
plt.show()

# As we can see from the plot duration is a very important feature. 
# This is the duration of last call with client. If the call duration is more , there are higher chances of getting a yes from the client. It has been sorted in descending order. Succes , cellular, housing, unknown, campaign are also highly correlated
# Model Selection

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size

# count plot for deposit
#the dataset is fairly balanced
sns.countplot(data=df,x=df['y'])

# Drop the 'y' column if it's our target and not numeric.
# Select only numeric columns for scaling.
numeric_features = df.select_dtypes(include=[np.number])
if 'y' in numeric_features.columns:
    numeric_features = numeric_features.drop('y', axis=1)

# Initialize and fit the scaler
scaler = StandardScaler()
scaler.fit(numeric_features)
scaled_features = scaler.transform(numeric_features)

# Create a new DataFrame with scaled numeric features
df_feat = pd.DataFrame(scaled_features, columns=numeric_features.columns)
print(df_feat.head())

# Applying Logistic Regression
# Train Test Split
# 20% Data is set aside for tesing
X_train,X_test,y_train,y_test=train_test_split(scaled_features,df['y'],test_size=0.20, random_state=3)

param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 10000]
    }
]
logModel = LogisticRegression()

clf = GridSearchCV(logModel, param_grid = param_grid,scoring='accuracy', cv = 5 )
best_clf = clf.fit(X_train,y_train)


# Training Logistic Regression
# Finding Accuracy, AUC, False positive rate, True positive rate, confusion matrix and classificatio report

pred = best_clf.predict(X_test)
accLR = accuracy_score(y_test, pred)
y_pred_prob = best_clf.predict_proba(X_test)
aucScoreLR = roc_auc_score(y_test,  y_pred_prob[:,1])
fprLR, tprLR, thresholds = roc_curve(y_test, y_pred_prob[:,1] )
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print("AUC score for LR is ",aucScoreLR)
print("Test Accuracy score for LR is ",accuracy_score(y_test, pred))
predT=best_clf.predict(X_train)
print("Train Accuracy score for LR is ",accuracy_score(y_train, predT))
print("Best parameters for accuracy of LR are ",best_clf.best_params_)

# Training Logistic Regression for recall

clfR = GridSearchCV(logModel, param_grid = param_grid,scoring='recall', cv = 5 )
best_clfR = clfR.fit(X_train,y_train)

# print recall and best parameters
predR = best_clfR.predict(X_test)
predRT=best_clfR.predict(X_train)
recallLR=recall_score(y_test, predR)
print("Test Recall score for LR is ",recallLR)
print("Train recall score for LR is ",recall_score(y_train, predRT))
print("Best parameters for recall of LR are ",best_clfR.best_params_)

dfX=df.drop('y',axis=1)
dfX.head()
X_train,X_test,y_train,y_test=train_test_split(dfX,df['y'],test_size=0.20, random_state=3)

dt=DecisionTreeClassifier()
parameters={'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
            'min_samples_leaf':[1,2,3,4,5],
            'min_samples_split':[2,3,4,5],
            'criterion':['gini','entropy']}

clf = GridSearchCV(dt,parameters,scoring='accuracy',verbose=True)
best_clf = clf.fit(X_train,y_train)

# Training Decision Tree
# Finding Accuracy, AUC, False positive rate, True positive rate, confusion matrix and classificatio report
# get best parameters for retraining

pred = best_clf.predict(X_test)
accDT = accuracy_score(y_test, pred)
y_pred_prob = best_clf.predict_proba(X_test)
aucScoreDT = roc_auc_score(y_test,  y_pred_prob[:,1])
fprDT, tprDT, thresholds = roc_curve(y_test, y_pred_prob[:,1] )
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print("AUC score for Decision Tree is ",aucScoreDT)
print("Test Accuracy score for DT is ",accuracy_score(y_test, pred))
predT=best_clf.predict(X_train)
print("Train Accuracy score for DT is ",accuracy_score(y_train, predT))
print("Best parameters for DT are ",best_clf.best_params_)

#Gridsearchcv Training Decision Tree for recall

clfR = GridSearchCV(dt,parameters,scoring='recall',verbose=True)
best_clfR = clfR.fit(X_train,y_train)

predR = best_clfR.predict(X_test)
predRT=best_clfR.predict(X_train)
recallDT=recall_score(y_test, predR)
print("Test Recall score for DT is ",recallDT)
print("Train recall score for DT is ",recall_score(y_train, predRT))
print("Best parameters for recall of DT are ",best_clfR.best_params_)

# Applying KNN
# 20% Data is set aside for tesing
X_train,X_test,y_train,y_test=train_test_split(scaled_features,df['y'],test_size=0.20, random_state=3)

# trying different odd values of k for KNN and finding accuracy for them

knn = KNeighborsClassifier(n_neighbors=1)
accuracy_rate=[]
for i in range(1,40,2):
    knn=KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,X_train,y_train,cv=5)
    accuracy_rate.append(score.mean())
print(accuracy_rate)

# Plotting accuracy of KNN for every value of K. Accuracy is highest when K=37

plt.figure(figsize=(20,12))
plt.plot(range(1,40,2),accuracy_rate,color='green',linestyle='dashed',marker='o',markerfacecolor='yellow',markersize=10)
plt.title('Accuracy rate vs k value')
plt.xlabel('k')
plt.ylabel('Accuracy rate')

# Training KNN agaib for best value of K
# Finding Accuracy, AUC, False positive rate, True positive rate, confusion matrix and classificatio report

knn = KNeighborsClassifier(n_neighbors=37)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
y_pred_prob = knn.predict_proba(X_test)
aucScoreKNN = roc_auc_score(y_test,  y_pred_prob[:,1])
fprKNN, tprKNN, thresholds = roc_curve(y_test, y_pred_prob[:,1] )
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print("AUC score for KNN is ",aucScoreKNN)
accKNN = accuracy_score(y_test, pred)
print("Test Accuracy score for KNN is ",accuracy_score(y_test, pred))
predT=knn.predict(X_train)
print("Train Accuracy score for KNN is ",accuracy_score(y_train, predT))
#print("Best parameters for KNN are ",knn.best_params_)

# We implemented a K-Nearest Neighbors (KNN) model for Bank Marketing Classification Project
# Training KNN for different odd values of K to find maximum Recall

knn = KNeighborsClassifier()
recall_rate=[]
for i in range(1,40,2):
    knn=KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,X_train,y_train,cv=5,scoring='recall')
    recall_rate.append(score.mean())
print(recall_rate)

# Plotting recall of values of K

plt.figure(figsize=(20,12))
plt.plot(range(1,40,2),recall_rate,color='green',linestyle='dashed',marker='o',markerfacecolor='yellow',markersize=10)
plt.title('Recall of values of K vs k value')
plt.xlabel('k')
plt.ylabel('Recall of values of K')

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
recallKMM=recall_score(y_test, pred)
print("Test Recall score for KNN is ",recall_score(y_test, pred))
predT=knn.predict(X_train)
print("Train Recall score for KNN is ",recall_score(y_train, predT))
#print("Best parameters for KNN are ",knn.best_params_)

# Model Tuning and Validation
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# Hyperparameter tuning for Logistic Regression
param_grid_lr = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': np.logspace(-4, 4, 20),
    'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 1000, 2500, 10000]
}

grid_lr = GridSearchCV(LogisticRegression(), param_grid_lr, scoring='roc_auc', cv=3)
grid_lr.fit(X_train, y_train)

# Best hyperparameters for Logistic Regression
best_params_lr = grid_lr.best_params_
print("Logistic Regression - Best Parameters:", best_params_lr)

# Hyperparameter tuning for Decision Tree
param_grid_dt = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'min_samples_split': [2, 3, 4, 5],
    'criterion': ['gini', 'entropy']
}

grid_dt = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, scoring='roc_auc', cv=3)
grid_dt.fit(X_train, y_train)

# Best hyperparameters for Decision Tree
best_params_dt = grid_dt.best_params_
print("Decision Tree - Best Parameters:", best_params_dt)
# Hyperparameter tuning for KNN
param_grid_knn = {
    'n_neighbors': list(range(1, 31)),  # Adjust the range based on your preference
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Different algorithms for nearest neighbors search

}

grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, scoring='roc_auc', cv=3)
grid_knn.fit(X_train, y_train)

# Best hyperparameters for KNN
best_params_knn = grid_knn.best_params_
print("K-Nearest Neighbors - Best Parameters:", best_params_knn)
# Cross-validation for each model
lr_cv_scores = cross_val_score(LogisticRegression(**best_params_lr), X_train, y_train, cv=3, scoring='roc_auc')
dt_cv_scores = cross_val_score(DecisionTreeClassifier(**best_params_dt), X_train, y_train, cv=3, scoring='roc_auc')
knn_cv_scores = cross_val_score(KNeighborsClassifier(**best_params_knn), X_train, y_train, cv=3, scoring='roc_auc')
print("Logistic Regression - Cross-validation Scores:", lr_cv_scores)
print("Decision Tree - Cross-validation Scores:", dt_cv_scores)
print("K-Nearest Neighbors - Cross-validation Scores:", knn_cv_scores)

# Perform ROC curve analysis and calculate AUC-ROC for each model
lr_probs = grid_lr.predict_proba(X_test)[:, 1]
dt_probs = grid_dt.predict_proba(X_test)[:, 1]
knn_probs = grid_knn.predict_proba(X_test)[:, 1]

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
dt_fpr, dt_tpr, _= roc_curve(y_test, dt_probs)
knn_fpr, knn_tpr,_= roc_curve(y_test, knn_probs)

lr_auc = roc_auc_score(y_test, lr_probs)
dt_auc = roc_auc_score(y_test, dt_probs)
knn_auc = roc_auc_score(y_test, knn_probs)

print("AUC-ROC  for Logistic Regression:", lr_auc)
print("AUC-ROC  for Decision Tree:", dt_auc)
print("AUC-ROC  for KNN:", knn_auc)

# Visualize ROC curves
plt.figure(figsize=(10, 8))
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.2f})')
plt.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC = {dt_auc:.2f})')
plt.plot(knn_fpr, knn_tpr, label=f'KNN (AUC = {knn_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()

# Threshold Selection for Logistic Regression
lr_probs = grid_lr.predict_proba(X_test)[:, 1]
dt_probs = grid_dt.predict_proba(X_test)[:, 1]
knn_probs = grid_knn.predict_proba(X_test)[:, 1]
lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_test, lr_probs)
dt_fpr, dt_tpr, dt_thresholds= roc_curve(y_test, dt_probs)
knn_fpr, knn_tpr,knn_thresholds= roc_curve(y_test, knn_probs)
operating_point_index_lr = np.argmax(lr_tpr - lr_fpr)
threshold_selected_lr = lr_thresholds[operating_point_index_lr]
print("Selected Threshold for Logistic Regression:", threshold_selected_lr)

# Threshold Selection for Decision Tree
operating_point_index_dt = np.argmax(dt_tpr - dt_fpr)
threshold_selected_dt = dt_thresholds[operating_point_index_dt]
print("Selected Threshold for Decision Tree:", threshold_selected_dt)

# Threshold Selection for KNN
operating_point_index_knn = np.argmax(knn_tpr - knn_fpr)
threshold_selected_knn = knn_thresholds[operating_point_index_knn]
print("Selected Threshold for KNN:", threshold_selected_knn)

from sklearn.metrics import precision_recall_curve

# Calculate precision and recall for each models
precision_lr, recall_lr, thresholds_pr_lr = precision_recall_curve(y_test, lr_probs)
precision_dt, recall_dt, thresholds_pr_dt = precision_recall_curve(y_test, dt_probs)
precision_knn, recall_knn, thresholds_pr_knn = precision_recall_curve(y_test, knn_probs)

# Plot Precision-Recall curve for Logistic Regression
plt.plot(recall_lr, precision_lr, label='Logistic Regression')
# Mark the selected threshold
plt.scatter(recall_lr[np.argmax(thresholds_pr_lr >= threshold_selected_lr)], 
            precision_lr[np.argmax(thresholds_pr_lr >= threshold_selected_lr)], color='red')
# Add label for the selected threshold
plt.text(recall_lr[np.argmax(thresholds_pr_lr >= threshold_selected_lr)] + 0.05, 
         precision_lr[np.argmax(thresholds_pr_lr >= threshold_selected_lr)], 
         f'Threshold: {threshold_selected_lr:.2f}', fontsize=10)

# Decision Tree
plt.plot(recall_dt, precision_dt, label='Decision Tree')
plt.scatter(recall_dt[np.argmax(thresholds_pr_dt >= threshold_selected_dt)], 
            precision_dt[np.argmax(thresholds_pr_dt >= threshold_selected_dt)], color='red')
plt.text(recall_dt[np.argmax(thresholds_pr_dt >= threshold_selected_lr)] + 0.05, 
         precision_dt[np.argmax(thresholds_pr_dt >= threshold_selected_dt)], 
         f'Threshold: {threshold_selected_dt:.2f}', fontsize=10)

# KNN
plt.plot(recall_knn, precision_knn, label='KNN')
plt.scatter(recall_knn[np.argmax(thresholds_pr_knn >= threshold_selected_knn)], 
            precision_knn[np.argmax(thresholds_pr_knn >= threshold_selected_knn)], color='red')
plt.text(recall_knn[np.argmax(thresholds_pr_knn >= threshold_selected_knn)] + 0.05, 
         precision_knn[np.argmax(thresholds_pr_knn >= threshold_selected_knn)], 
         f'Threshold: {threshold_selected_knn:.2f}', fontsize=10)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve with Selected Thresholds')
plt.legend()
plt.grid(True)
plt.show()

# Make predictions using the selected threshold for each models
lr_preds_selected_threshold = (lr_probs >= threshold_selected_lr).astype(int)

dt_preds_selected_threshold = (dt_probs >= threshold_selected_lr).astype(int)

knn_preds_selected_threshold = (knn_probs >= threshold_selected_knn).astype(int)

# Print the first few predictions for each model
print("Classification Report for Logistic Regression:")
print(classification_report(y_test, lr_preds_selected_threshold))

# Generate classification report for Decision Tree predictions
print("\nClassification Report for Decision Tree:")
print(classification_report(y_test, dt_preds_selected_threshold))

# Generate classification report for KNN predictions
print("\nClassification Report for KNN:")
print(classification_report(y_test, knn_preds_selected_threshold))

# Final Model Selection (Select the best-performing model based on AUC-ROC)
from sklearn.metrics import f1_score, accuracy_score

lr_cv_scores = cross_val_score(grid_lr.best_estimator_, X_train, y_train, cv=5, scoring='roc_auc')
dt_cv_scores = cross_val_score(grid_dt.best_estimator_, X_train, y_train, cv=5, scoring='roc_auc')
knn_cv_scores = cross_val_score(grid_knn.best_estimator_, X_train, y_train, cv=5, scoring='roc_auc')

lr_precision = cross_val_score(grid_lr.best_estimator_, X_train, y_train, cv=3, scoring='precision').mean()
dt_precision = cross_val_score(grid_dt.best_estimator_, X_train, y_train, cv=3, scoring='precision').mean()
knn_precision = cross_val_score(grid_knn.best_estimator_, X_train, y_train, cv=3, scoring='precision').mean()

lr_f1_score = cross_val_score(grid_lr.best_estimator_, X_train, y_train, cv=3, scoring='f1').mean()
dt_f1_score = cross_val_score(grid_dt.best_estimator_, X_train, y_train, cv=3, scoring='f1').mean()
knn_f1_score = cross_val_score(grid_knn.best_estimator_, X_train, y_train, cv=3, scoring='f1').mean()

lr_recall = cross_val_score(grid_lr.best_estimator_, X_train, y_train, cv=5, scoring='recall').mean()
dt_recall = cross_val_score(grid_dt.best_estimator_, X_train, y_train, cv=5, scoring='recall').mean()
knn_recall = cross_val_score(grid_knn.best_estimator_, X_train, y_train, cv=5, scoring='recall').mean()

lr_precision = cross_val_score(grid_lr.best_estimator_, X_train, y_train, cv=5, scoring='precision').mean()
dt_precision = cross_val_score(grid_dt.best_estimator_, X_train, y_train, cv=5, scoring='precision').mean()
knn_precision = cross_val_score(grid_knn.best_estimator_, X_train, y_train, cv=5, scoring='precision').mean()

# Print evaluation results

models_auc = {'Logistic Regression': lr_auc, 'Decision Tree': dt_auc, 'KNN': knn_auc}
best_model = max(models_auc, key=models_auc.get)
print(f"The best-performing model based on AUC-ROC is: {best_model}")
# dictionary to store evaluation results for each model
evaluation_results = {
    'Logistic Regression': {
        'model':grid_lr,
        'AUC-ROC': lr_cv_scores.mean(),
        'Accuracy': accLR,
        'Precision': lr_precision,
        'Recall': lr_recall,
        'F1 score': lr_f1_score
        
    },
    'Decision Tree': {
        'model':grid_dt,
        'AUC-ROC': dt_cv_scores.mean(),
        'Accuracy': dt_recall,
        'Precision': dt_precision,
        'Recall': recallDT,
        'F1 score': dt_f1_score
        
    },
    'KNN': {
        'model':grid_knn,
        'AUC-ROC': knn_cv_scores.mean(),
        'Accuracy': accKNN,
        'Precision': knn_precision,
        'Recall': knn_recall,
        'F1 score': knn_f1_score
       
        
    }
}

# Print evaluation results for each model
for model, results in evaluation_results.items():
    print(model + ":")
    for metric, value in results.items():
        print(f"{metric}: {value}")
    print()

# Select the best-performing model based on the highest AUC-ROC
best_model_name = max(evaluation_results, key=lambda x: evaluation_results[x]['AUC-ROC'])
print("Best-performing model based on AUC-ROC:", best_model)

# Model Testing (Evaluate the selected model on the testing dataset)
# Get the actual best model object using the name
best_model = evaluation_results[best_model_name]['model']

# Calculate AUC-ROC score on the testing dataset
test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
print(f"AUC-ROC score on the testing dataset for the selected model ({best_model_name}): {test_auc:.2f}")

# Make predictions on the testing dataset
y_pred = best_model.predict(X_test)

# Evaluate the Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

# Define individual models
logistic_regression = grid_lr.best_estimator_
decision_tree = grid_dt.best_estimator_
knn = grid_knn.best_estimator_

# Initialize ensemble method (Voting Classifier)
voting_clf = VotingClassifier(
    estimators=[
        ('lr', logistic_regression),
        ('dt', decision_tree),
        ('knn', knn)
    ],
    voting='soft'  # Use soft voting for probability predictions
)
# Define the train-test split

# Train ensemble method
voting_clf.fit(X_train, y_train)

# Evaluate ensemble method
ensemble_auc = roc_auc_score(y_test, voting_clf.predict_proba(X_test)[:, 1])
print("AUC-ROC score for the ensemble method:", ensemble_auc)
# Get predicted probabilities for the positive class
lr_probs = logistic_regression.predict_proba(X_test)[:, 1]
dt_probs = decision_tree.predict_proba(X_test)[:, 1]
knn_probs = knn.predict_proba(X_test)[:, 1]

# Calculate AUC-ROC for each model
lr_auc = roc_auc_score(y_test, lr_probs)
dt_auc = roc_auc_score(y_test, dt_probs)
knn_auc = roc_auc_score(y_test, knn_probs)

print("AUC-ROC for Logistic Regression:", lr_auc)
print("AUC-ROC for Decision Tree:", dt_auc)
print("AUC-ROC for KNN:", knn_auc)
from sklearn.metrics import accuracy_score
print("Accuracy: ")
for clf in (logistic_regression, decision_tree, knn, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

print("Recall: ")
for clf in (logistic_regression, decision_tree, knn, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, recall_score(y_test, y_pred))

print("F1-score: ")
for clf in (logistic_regression, decision_tree, knn, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, f1_score(y_test, y_pred))

print("Precision: ")
for clf in (logistic_regression, decision_tree, knn, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, precision_score(y_test, y_pred))

# Identify Class Imbalance
class_distribution = y_train.value_counts()
print("Class Distribution:")
print(class_distribution)
from imblearn.over_sampling import SMOTE

# Apply SMOTE for oversampling the minority class
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train models on resampled data
grid_lr.fit(X_train_resampled, y_train_resampled)
grid_dt.fit(X_train_resampled, y_train_resampled)
grid_knn.fit(X_train_resampled, y_train_resampled)

# Evaluate models on test data
lr_auc_resampled = roc_auc_score(y_test, grid_lr.predict_proba(X_test)[:, 1])
dt_auc_resampled = roc_auc_score(y_test, grid_dt.predict_proba(X_test)[:, 1])
knn_auc_resampled = roc_auc_score(y_test, grid_knn.predict_proba(X_test)[:, 1])

print("AUC-ROC score after dealing with class imbalance (Logistic Regression):", lr_auc_resampled)
print("AUC-ROC score after dealing with class imbalance (Decision Tree):", dt_auc_resampled)
print("AUC-ROC score after dealing with class imbalance (KNN):", knn_auc_resampled)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

lr_pred = grid_lr.predict(X_test)
dt_pred = grid_dt.predict(X_test)
knn_pred = grid_knn.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)

lr_precision = precision_score(y_test, lr_pred)
dt_precision = precision_score(y_test, dt_pred)
knn_precision = precision_score(y_test, knn_pred)

lr_recall = recall_score(y_test, lr_pred)
dt_recall = recall_score(y_test, dt_pred)
knn_recall = recall_score(y_test, knn_pred)

lr_f1 = f1_score(y_test, lr_pred)
dt_f1 = f1_score(y_test, dt_pred)
knn_f1 = f1_score(y_test, knn_pred)

# Print evaluation metrics
print("Accuracy:")
print("Logistic Regression:", lr_accuracy)
print("Decision Tree:", dt_accuracy)
print("KNN:", knn_accuracy)
print("\nPrecision:")
print("Logistic Regression:", lr_precision)
print("Decision Tree:", dt_precision)
print("KNN:", knn_precision)
print("\nRecall:")
print("Logistic Regression:", lr_recall)
print("Decision Tree:", dt_recall)
print("KNN:", knn_recall)
print("\nF1-score:")
print("Logistic Regression:", lr_f1)
print("Decision Tree:", dt_f1)
print("KNN:", knn_f1)

# Example of feature engineering (polynomial features)
from sklearn.preprocessing import PolynomialFeatures
                       
poly = PolynomialFeatures(degree=1)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
# Train models on polynomial features
grid_lr.fit(X_train_poly, y_train)
grid_dt.fit(X_train_poly, y_train)
grid_knn.fit(X_train_poly, y_train)

# Evaluate models on test data
# Before Feature Engineering
# Train models using original features
X_test_poly = poly.transform(X_test)  # Assuming 'poly' is the polynomial features transformer fitted on the training data

# Verify data consistency between training and test data
assert X_train_poly.shape[1] == X_test_poly.shape[1], "Number of features in training and test data are different after feature engineering"

# Evaluate models before feature engineering

# After Feature Engineering

# Evaluate models after feature engineering
lr_auc_after = roc_auc_score(y_test, grid_lr.predict_proba(X_test_poly)[:, 1])
dt_auc_after = roc_auc_score(y_test, grid_dt.predict_proba(X_test_poly)[:, 1])
knn_auc_after = roc_auc_score(y_test, grid_knn.predict_proba(X_test_poly)[:, 1])

# Compare AUC-ROC scores before and after feature engineering

print("AUC-ROC after feature engineering (Logistic Regression):", lr_auc_after)

print("AUC-ROC after feature engineering (Decision Tree):", dt_auc_after)

print("AUC-ROC after feature engineering (KNN):", knn_auc_after)

# You can similarly compare other evaluation metrics such as accuracy, precision, recall, and F1-score