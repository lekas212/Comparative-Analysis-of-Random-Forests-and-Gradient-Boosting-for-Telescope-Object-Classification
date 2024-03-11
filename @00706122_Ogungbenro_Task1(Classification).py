#!/usr/bin/env python
# coding: utf-8

# # Telescope Prediction Using Machine learning algorithm

# # DATA PREPROCESSING

# In[1]:


# importing libraries
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


# In[86]:


# Loading the dataset
dataset = pd.read_csv('telescope_data.csv')


# In[87]:


dataset.shape


# In[88]:


dataset.head()


# In[89]:


dataset.tail()


# In[90]:


dataset.info()


# In[91]:


dataset['class'] = dataset['class'].map({'g': 0, 'h': 1})


# In[92]:


dataset.head()


# In[93]:


dataset.tail()


# In[94]:


dataset.info()


# In[95]:


# Check for missing values
missing_values = dataset.isnull().sum()


# In[96]:


missing_values


# In[97]:


dataset.describe()


# In[98]:


dataset.describe(include='all')


# # EXPLORATORY DATA ANALYSIS

# In[99]:


# Assuming your dataset is named dataset
class_counts = dataset['class'].value_counts()

# Plotting a pie chart
plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
plt.title("Distribution of 'class'")
plt.show()


# In[101]:


# Plot pie chart
plt.figure(figsize=(4, 4))
y_resampled.value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'salmon'])
plt.title("Distribution of 'class' after SMOTE")
plt.show()


# In[102]:


import seaborn as sns
import matplotlib.pyplot as plt

# Select relevant columns for correlation
correlation_data = dataset[['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']]

# Calculate the correlation matrix
correlation_matrix = correlation_data.corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[103]:


# Explore the relationship between two numerical features 
sns.scatterplot(x='fConc', y='fConc1', data=dataset, hue='class')
plt.title('Scatterplot of fConc vs. fConc1')
plt.show()


# In[104]:


# Explore the relationship between two numerical features (e.g., 'fLength' and 'fWidth')
sns.scatterplot(x='fLength', y='fWidth', data=dataset, hue='class')
plt.title('Scatterplot of fLength vs. fWidth')
plt.show()


# In[105]:


# Explore pair-wise relationships between numerical features
sns.pairplot(dataset, hue='class', vars=['fLength', 'fWidth', 'fSize', 'fConc'])
plt.title('Pairplot of Selected Features')
plt.show()


# In[106]:


dataset.head()


# # Data Balancing using SMOTE

# In[100]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Assuming X contains your features and y contains your target variable
X = dataset.drop('class', axis=1)
y = dataset['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to over-sample the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Now, X_resampled and y_resampled contain the balanced dataset


# # THE APPLICATION OF DATA MINING TECHNIQUES
# 

# # Random Forest

# In[108]:


# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100,random_state=42)


# In[109]:


# Fit the classifier on the resampled data
rf_classifier.fit(X_resampled, y_resampled)


# In[110]:


# Make predictions on the test set
y_pred_rf = rf_classifier.predict(X_test)


# In[111]:


y_pred_rf


# In[112]:


y_test


# In[113]:


# Evaluate the performance
print("Random Forests:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))


# In[114]:


# Confusion Matrix for Random Forests
cm_rf = confusion_matrix(y_test, y_pred_rf)


# In[125]:


# Plotting Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# # Standardization

# In[116]:


# Standardize the features
sc = StandardScaler()

# Fit the scaler to the resampled data and transform it
X_train_s = sc.fit_transform(X_resampled)

# Transform the test data using the same scaler
X_test_s = sc.transform(X_test)


# In[117]:


# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100,random_state=42)


# In[118]:


# Fit the classifier on the resampled data
rf_classifier.fit(X_train_s, y_resampled)


# In[119]:


y_pred_r = rf_classifier.predict(X_test_s)


# In[120]:


y_pred_r


# In[121]:


y_test


# In[122]:


# Evaluate the performance
print("Random Forests:")
print("Accuracy:", accuracy_score(y_test, y_pred_r))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_r))
print("Classification Report:\n", classification_report(y_test, y_pred_r))


# In[124]:


# Confusion Matrix for Random Forests
cm_rf = confusion_matrix(y_test, y_pred_rf)


# In[39]:


# Plotting Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title('Confusion Matrix - Random Forests')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# # Gradient Boost

# In[41]:


# Initialize the Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)


# In[42]:


# Train the model
gb_classifier.fit(X_resampled, y_resampled)


# In[43]:


# Make predictions on the test set
y_pred_gb = gb_classifier.predict(X_test)


# In[44]:


y_pred_gb


# In[45]:


y_test


# In[46]:


# Evaluate the performance
print("\nGradient Boosting:")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gb))
print("Classification Report:\n", classification_report(y_test, y_pred_gb))


# In[126]:


# Confusion Matrix for Gradient Boost
cm_gb_g = confusion_matrix(y_test, y_pred_gb)


# In[127]:


# Plotting Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm_gb_g, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title('Confusion Matrix - Gradient Boosting')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[47]:


# Initialize the StandardScaler
sc = StandardScaler()

# Fit the scaler to the training data and transform it
X_train_s = sc.fit_transform(X_resampled)

# Transform the test data using the same scaler
X_test_s = sc.transform(X_test)


# In[48]:


# Initialize the Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)


# In[49]:


# Train the model
gb_classifier.fit(X_train_s, y_resampled)


# In[50]:


# Make predictions on the test set
y_pred_gb = gb_classifier.predict(X_test_s)


# In[51]:


y_pred_gb


# In[52]:


y_test


# In[53]:


# Evaluate the performance
print("\nGradient Boosting:")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gb))
print("Classification Report:\n", classification_report(y_test, y_pred_gb))


# In[55]:


# Confusion Matrix for Gradient Boost
cm_gb = confusion_matrix(y_test, y_pred_gb)


# In[56]:


cm_gb


# In[57]:


# Plotting Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title('Confusion Matrix - Gradient Boosting')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# # HYPERPARAMETER TUNING

# In[59]:


# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100,],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}



# In[60]:


# Create the Random Forest classifier
rf_classifier = RandomForestClassifier()


# In[61]:


# Use GridSearchCV to perform grid search
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_s, y_resampled)


# In[62]:


# Get the best hyperparameters
best_params = grid_search.best_params_


# In[63]:


# Print the best hyperparameters
print("Best Hyperparameters:")
print(best_params)


# In[64]:


# Use the best hyperparameters to create the final model
final_rf_model = RandomForestClassifier(**best_params)


# In[65]:


# Fit the model to the entire training set
final_rf_model.fit(X_train_s, y_resampled)


# In[66]:


# Make predictions on the test set
y_pred_test = final_rf_model.predict(X_test_s)


# In[67]:


y_pred_test


# In[68]:


y_test


# In[69]:


# Evaluate the performance
print("Random Forests:")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print("Classification Report:\n", classification_report(y_test, y_pred_test))


# In[130]:


cm_gb_tests = confusion_matrix(y_test, y_pred_test)


# In[132]:


# Plotting Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm_gb_tests, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[70]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier


# In[71]:


# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100,],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}


# In[72]:


# Initialize the Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(random_state=42)


# In[73]:


# Use GridSearchCV to perform grid search
grid_search_gb = GridSearchCV(estimator=gb_classifier, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)


# In[74]:


grid_search_gb.fit(X_train_s, y_resampled)


# In[75]:


# Get the best hyperparameters
best_params_gb = grid_search_gb.best_params_


# In[76]:


# Print the best hyperparameters
print("Best Hyperparameters for Gradient Boosting:")
print(best_params_gb)


# In[77]:


# Use the best hyperparameters to create the final Gradient Boosting model
final_gb_model = GradientBoostingClassifier(**best_params)


# In[78]:


# Fit the model to the entire training set
final_gb_model.fit(X_train_s, y_resampled)


# In[79]:


# Make predictions on the test set
y_pred_test_gb = final_gb_model.predict(X_test_s)


# In[80]:


y_pred_test_gb


# In[81]:


y_test


# In[83]:


# Evaluate the performance
print("Gradient Boost:")
print("Accuracy:", accuracy_score(y_test, y_pred_test_gb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test_gb))
print("Classification Report:\n", classification_report(y_test, y_pred_test_gb))


# In[128]:


# Confusion Matrix for Random Forests
cm_gb_test = confusion_matrix(y_test, y_pred_test_gb)


# In[129]:


# Plotting Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm_gb_test, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title('Confusion Matrix - Gradient Boosting')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[134]:


# Create a list of algorithm names and their corresponding accuracy values
algorithms = ['Random Forests', 'Gradient Boosting']
accuracies = [accuracy_score(y_test, y_pred_r), accuracy_score(y_test, y_pred_gb)]

# Create a bar plot to display the accuracy of each algorithm
plt.figure(figsize=(10, 6))
plt.bar(algorithms, accuracies, color=['blue', 'green'])
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Random Forests and Gradient Boosting')
plt.ylim(0.0, 1.0)

# Display the accuracy values on top of the bars
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')

plt.show()


# In[ ]:




