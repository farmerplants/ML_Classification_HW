# UMN Fintech -- Machine Learning &amp; Classification Homework
## The solutions can be found in the Resources and Solutions folder:
#### Credit Risk Resampling Techniques: https://github.com/farmerplants/ML_Classification_HW/blob/main/Resources%20and%20Solutions/credit_risk_resampling.ipynb
#### Ensemble Learning: https://github.com/farmerplants/ML_Classification_HW/blob/main/Resources%20and%20Solutions/credit_risk_ensemble.ipynb

## Homework Background:
### Mortgages, student and auto loans, and debt consolidation are just a few examples of credit and loans that people seek online. Peer-to-peer lending services such as Loans Canada and Mogo let investors loan people money without using a bank. However, because investors always want to mitigate risk, a client has asked that you help them predict credit risk with machine learning techniques.
### In this assignment you will build and evaluate several machine learning models to predict credit risk using data you'd typically see from peer-to-peer lending services. Credit risk is an inherently imbalanced classification problem (the number of good loans is much larger than the number of at-risk loans), so you will need to employ different techniques for training and evaluating models with imbalanced classes. You will use the imbalanced-learn and Scikit-learn libraries to build and evaluate models using resampling techniques and ensemble learning.

## Homework Instructions:
### Resampling:
#### Use the imbalanced learn library to resample the LendingClub data and build and evaluate logistic regression classifiers using the resampled data. To begin:
#### - Read the CSV into a DataFrame.
#### - Split the data into Training and Testing sets.
#### - Scale the training and testing data using the StandardScaler from sklearn.preprocessing.
#### Use the provided code to run a Simple Logistic Regression:
#### - Fit the logistic regression classifier.
#### - Calculate the balanced accuracy score.
#### - Display the confusion matrix.
#### - Print the imbalanced classification report.
#### Next you will:
#### - Oversample the data using the Naive Random Oversampler and SMOTE algorithms.
#### - Undersample the data using the Cluster Centroids algorithm.
#### - Over- and undersample using a combination SMOTEENN algorithm.
#### For each of the above, you will need to:
#### - Train a logistic regression classifier from sklearn.linear_model using the resampled data.
#### - Calculate the balanced accuracy score from sklearn.metrics.
#### - Display the confusion matrix from sklearn.metrics.
#### - Print the imbalanced classification report from imblearn.metrics.
#### Use the above to answer the following questions:
#### - Which model had the best balanced accuracy score?
#### - Which model had the best recall score?
#### - Which model had the best geometric mean score?

### Ensemble Learning:
#### In this section, you will train and compare two different ensemble classifiers to predict loan risk and evaluate each model. You will use the Balanced Random Forest Classifier and the Easy Ensemble Classifier. Refer to the documentation for each of these to read about the models and see examples of the code. To begin:
#### - Read the data into a DataFrame using the provided starter code.
#### - Split the data into training and testing sets.
#### - Scale the training and testing data using the StandardScaler from sklearn.preprocessing.
#### Then, complete the following steps for each model:
#### - Train the model using the quarterly data from LendingClub provided in the Resource folder.
#### - Calculate the balanced accuracy score from sklearn.metrics.
#### - Display the confusion matrix from sklearn.metrics.
#### - Generate a classification report using the imbalanced_classification_report from imbalanced learn.
#### - For the balanced random forest classifier only, print the feature importance sorted in descending order (most important feature to least important) along with the feature score.
#### Use the above to answer the following questions:
#### - Which model had the best balanced accuracy score?
#### - Which model had the best recall score?
#### - Which model had the best geometric mean score?
#### - What are the top three features?





## Conclusions:
### Resampling: 
#### Which model had the best balanced accuracy score?
#### - SMOTEEN (combined over/undersampling) and SMOTE both had the best score at 0.9947. ClusterCentroids had a score of 0.9933.
#### Which model had the best recall score?
#### - All three models had a recall average of 0.99, however, the SMOTE and SMOTEENN models both had a recall score of 0.98 on high_risk, while ClusterCentroids had a score of 0.99.
#### Which model had the best geometric mean score?
#### - All three models had the same geo score at 0.99.
### Ensemble Learning:
#### Which model had the best balanced accuracy score?
#### - The EasyEnsembleClassifier model had the best score at 0.9333, as opposed to the BalancedRandomForestClassifier model with a score of 0.8109.
#### Which model had the best recall score?
#### - The EasyEnsembleClassifier model once again had the best average recall score at 0.95, while the BRFC model had an average score of 0.90.
#### Which model had the best geometric mean score?
#### - Again, the EasyEnsembleClassifier model had the best score at 0.93, compared to the BRFC model's score of 0.81.
#### What are the top three features?
#### - 1 - total_rec_pmcp (0.07549); 2 - total_pymnt (0.06563); 3 - last_pymnt_amt (0.06129)
