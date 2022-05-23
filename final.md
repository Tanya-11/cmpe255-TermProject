---
title: Detecting Phishing Websites
date: "May 2022"
author: Reshma Parakkal, Santhosh Bodla, Surbhi Dogra, Tanya Gupta, San José State University

header-includes: |
  \usepackage{booktabs}
  \usepackage{caption}
---

# Abstract

Phishing is a dangerous threats to your online accounts and data, because it peretends to be a legitimate source, and incorporates social engineering to make victims
to fall for the attack. In order to prevent phishing attack we aim  to predict the data based on the URL of the website. In this project first we have preformed data preprocessing to clean the data
and then impute the missing values. Later the Discovery of malicious websites is performed by using algorithms like logistic regression, XGBoost, Random Forest and their performance will be evaluated
before and after parameter tuning.

# Introduction
Phishing is a technique used by phishers to steal the user’s confidential and sensitive data, which could lead to financial loss for any individual or organization. Phishing is a serious security problem in the cyber-world and is an illegal act done by the attackers to steal personal identity data and financial account credentials from web users. Phishing sites lure victims to surf on a fake website and steal their confidential data. Due to this reason, research has been conducted to detect and prevent phishing attacks, where models are constructed for performing classification on phishing websites.
We will be incorporating classification models, whose input will be URL attributes. This model will be trained with the exhaustive dataset to maximum accuracy. The dataset includes two dataset with 58,645 and 88,647 website’s URL deemed as fraudulent or real. The website is identified by its attributes which are fed to classification models for classification.

   
# Data Description
The data used has been gathered to develop and analyze different classifiers for detecting phishing websites using URL characteristics, URL resolving metrics and external services. Six groups of 
attributes can be found in the prepared dataset:\
  • Item attributes based on the whole URL properties.\
  • Item attributes based on the domain properties\
  • Item attributes based on the URL directory properties.\
  • Item attributes based on the URL file properties.\
  • Item attributes based on the URL parameter properties.\
  • Item attributes based on the URL resolving data and external metrics.
  
 As evident from picture below, the first group is based on the values of the characteristics on the entire URL string, but the values of the next four groups are based on sub-strings. The
last set of attributes is based on URL's resolve metrics as well as external services like Google's search index.

![image](https://user-images.githubusercontent.com/90728105/167979436-fcd7ba8f-8a1d-4a4f-9162-90fcc1443008.png)

This dataset has 111 features in total, except the target phishing attribute, which indicates if the instance is legitimate (0) or phishing (1). the dataset has two versions of the 
dataset, one with 58,645 occurrences with a more or less balanced balance between the target groups, with 30,647 instances categorized as phishing websites and 27,998 
instances labeled as legitimate. The second dataset has 88,647 cases, with 30,647 instances tagged as phishing and 58,000 instances identified as valid, with the goal of simulating a 
real-world situation in which there are more legitimate websites present. We have used dataset small for further analysis and model building as it has more balanced classes.
 
 ![image](https://user-images.githubusercontent.com/90728105/167975120-764f9474-f59b-4044-ab89-a44287ca6433.png)
 
 ## Models to be used
### Logistic Regression

Logistic regression is a supervised learning technique. Logistic Regression is used in statistical software to understand the relationship between the dependent variable and one or more independent variables by estimating probabilities using a logistic regression equation. Logistic Regression is used to calculate or predict the probability of a binary event occurring where the outcome can be either yes or no. In our case, in the dataset we are using, we need to predict based on the field values of some websites provided if the that particular website is a phishing website or a malicious one or not. So, the usecase in this case is again binary. Additionally, the training data we're using is independent of each other but at the same time can be linearly related and is of fairly large size. All these factors/ assumptions of the traning data are satisfied for implementing logistic regression.

### Random Forest Classifier
Random Forests is a classification and regression strategy that uses an ensemble approach. During training, the Random Forest classifier creates a number of decision trees and produces a class that is the mode of the classification classes of the individual trees. Because it uses a forest of classification trees to make a judgment, Random Forest classification outperforms all other decision tree methods. We used random forests in our method to increase the overall accuracy of the system because they accept missing values well. Our training dataset contains 58,645 and 88,647 website URLs that were classified as fake or real, respectively.

Consider the following scenario: the input is a PayPal phishing URL. The phishing site's keywords are retrieved and entered the Google search engine. If the top 10 Google results contain the exact page we're looking for, the isomorphic match will be successful, and the suspected URL will be judged real. Assume that the top 10 search results do not contain the page that we are looking for. Isomorphic comparison will fail with all ten outcomes in this situation. We can't just declare the suspicious URL is a phishing one at this point. This can happen if the keyword vector was unable to produce the best potential search results. We use the Random Forest classifier to minimize false positives because the feature vector is reliant on the domain but not on a single page within the domain. Although a PayPal website can have numerous pages, elements such as the domain name, IP address, creation date, Name servers, and so on are consistent across the PayPal domain and do not change from page to page. This gives us the confidence to proceed to the next step, in which the random forest classifier may make a judgment based on the feature vector. This also allows our method to predict the potential phishing target, which in this case is Paypal.com.

### XGBoost, or Extreme Gradient Boosting
It is a decision tree-based machine learning algorithm that improves performance through a process known as boosting. Gradient Boosting Decision Trees (GBDT) is a decision tree ensemble learning algorithm for classification and regression that is similar to random forest. To create a better model, ensemble learning techniques combine different machine learning algorithms, basically GBDT’s train an ensemble of shallow decision trees iteratively, with each iteration using the prior model's error residuals to fit the next model. The final prediction is a weighted sum of all of the tree predictions. In general Random forest “bagging” minimizes the variance and overfitting, while GBDT “boosting” minimizes the bias and underfitting.

XGBoost is a scalable and highly accurate version of gradient boosting that pushes the limits of computing power for boosted tree algorithms. It was designed primarily to increase machine learning model performance and computational speed. With XGBoost, trees are built in parallel, instead of sequentially like GBDT. It follows a level-wise strategy, scanning across gradient values and using these partial sums to evaluate the quality of splits at every possible split in the training set.

In our case, For predicting the maliciousness of websites, To get the best optimal results, we tune the XGB using a random search for fitting 3 folds for each of the 100 candidates totaling 300 fits. The optimal parameters obtained are subsample: 0.1, estimators: 500, min child weight: 1, max depth: 5, eta: 0.05, colsample bytree: 0.1. Using these parameters in the XGBClassifier gives us the feature importance of the data frame.

# Methods

## Data Preprocessing
Data Preprocessing is referred to as manipulation or dropping of data before it is used to ensure or enhance performance. It is basically the process of transforming the raw data into understandable format. Data preprocessing is the most important phase of machine learning. It includes removing irrelevant and redundant information from the data. Examples of data preprocessing include cleaning, instance selection, normalization, feature extraction and selection. The product of data preprocessing is the final training set. The dataset that we have selected contained irrelevant and meaningless information which has been removed.

Next, we analysed that the dataset contained '-1' values throughout where almost all the rows had this value, so we cannot drop all this data. We then checked the percentage of '-1' values in each column.As per the previous analysis, we have noticed that almost 80% of the dataset contains ‘-1’. Since most of the columns have ‘-1’, it would not be wise to remove them altogether as they may significantly affect the result. To tackle this, we remove the columns with less than 80% ‘-1’ and replace them with Nan. To improve the efficiency while testing and training, we drop the rest of the columns.

Once we have dropped values containing ‘-1’, the next step is to look at the missing values. There are three main reasons why values could be missing – Missing at random, Missing Completely at random, Not Missing at random. The initial approach initiated for imputing is using mean imputation. As the name suggests, the mean is calculated for the available values and replaced with the non-missing value’s number. An essential step to bear in mind during mean imputation is to remove outliers to prevent seeing absurd or surprising values as mean.

In addition to this, the missing values are also imputed using median and mode. These methods do not necessarily worry about outliers as they work using the middle values that are present when the column values are sorted. The last imputation method used is the most effective in predicting the missing values. It uses the Nearest Neighbor method known as KNN imputation, where the Nan values are replaced with the values of the neighboring values.

## Methods Followed
Data Preprocessing is referred to as manipulation or dropping of data before it is used to ensure or enhance performance. It is basically the process of transforming the raw data into understandable format. It includes removing irrelevant and redundant information from the data.The product of data preprocessing is the final training set. The dataset that we have selected contained irrelevant and meaningless information which has been removed.
Firstly, we filtered the data by dropping duplicate rows. We dropped 1653 rows in the dataset. These values were removed to reduce the dimensionality.



####Figure 3: Removing Duplicate Rows

Next, we analyzed that the dataset contained '-1' values throughout where almost all the rows had this value, so we cannot drop all this data. We then checked the percentage of '-1' values in each column.As per the previous analysis, we have noticed that almost 80% of the dataset contains ‘-1’. Since most of the columns have ‘-1’, it would not be wise to remove them altogether as they may significantly affect the result. To tackle this, we remove the columns with less than 80% ‘-1’ and replace them with Nan. To improve the efficiency while testing and training, we drop the rest of the columns.
We then visualized the missing data in the dataframe using the missing number library. The figure below shows the visualization of missing data after imputing Nan and it can be noted that a lot of params are missing.

####Figure 4 : Visualization of Missing Data
 
Once we have dropped values containing ‘-1’, the next step is to look at the missing values. There are three main reasons why values could be missing – Missing at random, Missing Completely at random, Not Missing at random. The initial approach initiated for imputing it using mean imputation. As the name suggests, the mean is calculated for the available values and replaced with the non-missing value’s number. An essential step to bear in mind during mean imputation is to remove outliers to prevent seeing absurd or surprising values as mean.
In addition to this, the missing values are also imputed using median and mode. These methods do not necessarily worry about outliers as they work using the middle values that are present when the column values are sorted. The last imputation method used is the most effective in predicting the missing values. It uses the Nearest Neighbor method known as KNN imputation, where the Nan values are replaced with the values of the neighboring values.

### A.Mean Imputed Data Analysis:
The data is missing at random in our Dataset, and there is no correlation with other variables  in the dataset. So, the strategy here is to use Mean Imputation to impute the Null or NAN values. 
We used the SimpleImputer class to accomplish this. SimpleImputer is a scikit-learn class for dealing with missing data in a predictive model dataset. 
The mean of the corresponding column is used to replace all missing data.
The data is then shown to see if there are any Null values. It is clear from the Bar Plot below that there are no more Null values.



### B.KNN Imputed Data Analysis
The Nan values or the null data is imputed using KNN imputer with nearest neighbor - 7 and the distance measure used is euclidean distance. 
The stratified split size of test data to train data is 25%.
Data is standardized using StandardScalar() function.
We have used 2 classifiers for analyzing the data : Logistic Regression and Random Forest Classifier.
We divided the data into categorical and numerical columns for better analysis.
Below is the visualization of  categorical in knn imputed data.


####Figure 5 : Visualization of categorical data in knn imputed data

## Modeling Comparisons
The accuracy is obtained using various models. The first two models were trained on Mean imputed data while the other models were trained on KNN imputed data. The following are the models:\

### Logistic Regression:
It is used in statistical software to understand the relationship between the dependent variable and one or more independent variables by estimating probabilities using a logistic regression equation. The accuracy found through Logistic Regression is 99.99 percent via Mean imputed data. Hence, it can be understood that this model is overfitting.\

### KNN Classifier:
The k-nearest neighbors (KNN) algorithm is a simple, supervised machine learning algorithm that can be used to solve both classification and regression problems. The accuracy found in the KNeighbour classifier is 95.34 percent via Mean imputed data. Below, the figure shows us the accuracy of the KNN Classifier model. \
![image](https://user-images.githubusercontent.com/90728105/169725911-df1bcd4c-9789-429c-8d2b-3de7ce23928d.png)

### XGBoost Classifier:
XGBoost is an algorithm that has recently been dominating applied machine learning and Kaggle competitions for structured or tabular data. XGBoost is an implementation of gradient boosted decision trees designed for speed and performance. The accuracy found through XGB CLassifier is 92.77 percent via KNN imputation. The performance metrics of XGBoost Classifier can be seen in the figure below- \
![image](https://user-images.githubusercontent.com/90728105/169725626-91195074-df1f-4746-9202-77c6caf3c950.png)

### Logistic Regression:
It is used in statistical software to understand the relationship between the dependent variable and one or more independent variables by estimating probabilities using a logistic regression equation. The accuracy found through Logistic Regression is 89.07 percent via KNN imputation. Performance metrics of Logistic Regression can be seen in the figure below- \
![image](https://user-images.githubusercontent.com/90728105/169725560-b18fe537-561b-4bd7-b9fc-c586ca641c45.png)

### Random Forest Classifier:
When a large number of decision tree operate as an ensemble, they make up Random Forest. Each tree in the random forest produces a class prediction, and the class with the most votes becomes the prediction of our model. The accuracy found through Random Forest is 95.63 percent via KNN imputation. The Performance metrics of Random Forest Classifier can be seen in the figure below- \
![image](https://user-images.githubusercontent.com/90728105/169725726-2876dff8-3258-489d-b5bb-d6ce9e899b71.png)


