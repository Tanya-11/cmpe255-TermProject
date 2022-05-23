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
  
 As evident from picture below, the first group is based on the values of the characteristics on the entire URL string, but the values of the next four groups are based on sub-strings. The last set of attributes is based on URL's resolve metrics as well as external services like Google's search index.

![image](https://user-images.githubusercontent.com/90728105/167979436-fcd7ba8f-8a1d-4a4f-9162-90fcc1443008.png)
#### Figure 1: URL attaributes 

This dataset has 111 features in total, except the target phishing attribute, which indicates if the instance is legitimate (0) or phishing (1). the dataset has two versions of the 
dataset, one with 58,645 occurrences with a more or less balanced balance between the target groups, with 30,647 instances categorized as phishing websites and 27,998 
instances labeled as legitimate. The second dataset has 88,647 cases, with 30,647 instances tagged as phishing and 58,000 instances identified as valid, with the goal of simulating a 
real-world situation in which there are more legitimate websites present. We have used dataset small for further analysis and model building as it has more balanced classes.
 
 ![image](https://user-images.githubusercontent.com/90728105/167975120-764f9474-f59b-4044-ab89-a44287ca6433.png)
 #### Figure 2: Data set Distribution based on phishing and non-phishing website
 
## Models

### Logistic Regression
Logistic Regression was used because it matched with our objective that we want to label the data into malicious or not. And, it had to be Binomial because we only had two labels i.e. phishing(1) or not(0). And, it’s fast since while doing data imputation it took longer time for imputing the values so I ensured that the model used should be fast. As per my research, I found that Logistic Regression is transparency. Logistic regression is called the “white box”. It’s easier to know why a set of parameters were labeled as phishing or legit.
An analyst may want to know “why the prediction works” or the need to restrict the equation from using certain data in specific ways. The classical example of drownings and ice cream sales being correlated together because more people both swim and drown in the summertime and also eat more ice cream in the summer. Ice cream sales might help indicate “when people will drown”, but it’s not going to indicate “why people are drowning”. The need to know “why” means that it’s important to restrict the ways data is used and assure logical inference. The more convoluted the formula and the less involved the analyst is the less understanding what caused what or why a prediction works and when it might stop working


### Random Forest Classifier
Random Forests is a classification and regression strategy that uses an ensemble approach. During training, the Random Forest classifier creates a number of decision trees and produces a class that is the mode of the classification classes of the individual trees. Because it uses a forest of classification trees to make a judgment, Random Forest classification outperforms all other decision tree methods. We used random forests in our method to increase the overall accuracy of the system because they accept missing values well. Our training dataset contains 58,645 and 88,647 website URLs that were classified as fake or real, respectively.

Consider the following scenario: the input is a PayPal phishing URL. The phishing site's keywords are retrieved and entered the Google search engine. If the top 10 Google results contain the exact page we're looking for, the isomorphic match will be successful, and the suspected URL will be judged real. Assume that the top 10 search results do not contain the page that we are looking for. Isomorphic comparison will fail with all ten outcomes in this situation. We can't just declare the suspicious URL is a phishing one at this point. This can happen if the keyword vector was unable to produce the best potential search results. We use the Random Forest classifier to minimize false positives because the feature vector is reliant on the domain but not on a single page within the domain. Although a PayPal website can have numerous pages, elements such as the domain name, IP address, creation date, Name servers, and so on are consistent across the PayPal domain and do not change from page to page. This gives us the confidence to proceed to the next step, in which the random forest classifier may make a judgment based on the feature vector. This also allows our method to predict the potential phishing target, which in this case is Paypal.com.

### XGBoost, or Extreme Gradient Boosting
It is a decision tree-based machine learning algorithm that improves performance through a process known as boosting. Gradient Boosting Decision Trees (GBDT) is a decision tree ensemble learning algorithm for classification and regression that is similar to random forest. To create a better model, ensemble learning techniques combine different machine learning algorithms, basically GBDT’s train an ensemble of shallow decision trees iteratively, with each iteration using the prior model's error residuals to fit the next model. The final prediction is a weighted sum of all of the tree predictions. In general Random forest “bagging” minimizes the variance and overfitting, while GBDT “boosting” minimizes the bias and underfitting.

The reason for choosing this classification was mainly due to its boosting ensemble technique. After doing research we found that boosting is greatly helpful in making a strong classifier  model from a combination of weak ones and is best for dealing with bias-variance tradeoff which helped us in avoiding memorizing the training set and also over-simplifying the model complexity. And, unlike KNN classifiers and Logistic Regression, it has inbuilt Cross-Validation, which is really helpful in increasing model accuracy. And, we mainly used for extracting important or relevant features from the dataset to avoid overfitting and redundancy could affect the accuracy and so that these features could be fed to classifiers like KNN and Logistic Regression who don’t suit best for such tasks. Since we didn’t have any categorical data except for the target instance, there were absolutely no issues in using XGBoost for boosting.

### KNN Classifier

KNN classifier was chosen for classification because it’s suited well for the dataset it’s labeled, has already been preprocessed to remove missing values and most importantly it was balanced, as KNN doesn’t perform well on imbalanced data. It classified the new data points based on the similarity measure of the earlier stored data points. For example, in the dataset of phishing and legit. KNN will store similar measures based on its attributes. When a new object comes it will check its similarity with those attributes itself. Hence, It helped us to classify the unseen data based on the similarity to the stored data that KNN classifiers store during the training phase.
For instance, for k = 5, and 3 of data points are ‘phishing’ and 2 are ‘non-phishing’, then the data point in question would be labeled ‘phishing’, since ‘phishing’ is the majority.
As KNN is prone to overfitting and achieve best accuracy, we did hypertuning of parameters using GridSearchCV to train our model multiple times on a range of parameters that we specified. That way, we could test our model with each parameter and figure out the optimal values to get the best accuracy results. And, used k-Fold Cross-Validation, which helped in increment of accuracy by 7% because it gave the model an opportunity to test on multiple splits so that we could get a better idea on how the model will perform on unseen data.


# Methods

## Data Preprocessing
Data Preprocessing is referred to as manipulation or dropping of data before it is used to ensure or enhance performance. It is basically the process of transforming the raw data into understandable format. Data preprocessing is the most important phase of machine learning. It includes removing irrelevant and redundant information from the data. Examples of data preprocessing include cleaning, instance selection, normalization, feature extraction and selection. The product of data preprocessing is the final training set. The dataset that we have selected contained irrelevant and meaningless information which has been removed.


After importing the dataset, preliminary work was done on it, including the preprocessing of the data to ensure the models aren’t trained on dirty data in terms of missing values , null values or irrelevant values or attributes. As an improperly trained model won’t actually be relevant. After carefully analyzing the data, first started with checking for null values in each attribute because they hold no significance. The 'is.na()' method has been used to check for missing data or null values. This approach goes over each and every column of the data set, looking for outliers with NA values that may have an impact on the calculation, and it was found that there were null values for each column. Then, checked for the 0 variance columns and dropped those columns, because they all have the same values and it means that the feature is constant and will not improve the performance of the model. Next, remove the rows with the same set of values in each row, as these wouldn’t have helped in the model's performance and also helped in reducing the length of data by 1653 rows. Furthermore, After doing a brief analysis, we found that the number of -1's in each feature is comparatively higher but as per the dataset description, the -1's values do not have any significance. URL attributes can never have negative values. For example, features like quantity, length, and params can never be negative. It is highly illogical to consider these negative values. These “-1” values are considered as missing values here. We calculated the percentage of “-1” values in each and every feature. Features that have more than 80 percent of their values as “-1” are dropped. Later, all the other features consisting of “-1” values are replaced with “NAN”. Then, the dataset was visualized using missingno library and still few missing values were found, which were imputed using mean, median and KNN  imputation. Though mean, median are famous imputation techniques, KNN imputation considers the relation among the attributes by replacing NAN with nearest neighbor estimated values, unlike mean or median who impute the value based only on column values. We decided to demonstrate the efficiency of KNN imputation by doing analysis on both mean and KNN imputed data.
The last imputation method used is the most effective in predicting the missing values. It uses the Nearest Neighbor method known as KNN imputation, where the Nan values are replaced with the values of the neighboring values.


## Methods Followed
Data Preprocessing is referred to as manipulation or dropping of data before it is used to ensure or enhance performance. It is basically the process of transforming the raw data into understandable format. It includes removing irrelevant and redundant information from the data.The product of data preprocessing is the final training set. The dataset that we have selected contained irrelevant and meaningless information which has been removed.
Firstly, we filtered the data by dropping duplicate rows. We dropped 1653 rows in the dataset. These values were removed to reduce the dimensionality.\
<img width="937" alt="Screen Shot 2022-05-22 at 7 08 17 PM" src="https://user-images.githubusercontent.com/25512807/169730371-8955ff44-6435-44ef-b532-2253e45d9678.png">
#### Figure 3: Removing Duplicate Rows


Next, we analyzed that the dataset contained '-1' values throughout where almost all the rows had this value, so we cannot drop all this data. We then checked the percentage of '-1' values in each column.As per the previous analysis, we have noticed that almost 80% of the dataset contains ‘-1’. Since most of the columns have ‘-1’, it would not be wise to remove them altogether as they may significantly affect the result. To tackle this, we remove the columns with less than 80% ‘-1’ and replace them with Nan. To improve the efficiency while testing and training, we drop the rest of the columns.
We then visualized the missing data in the dataframe using the missing number library. The figure below shows the visualization of missing data after imputing Nan and it can be noted that a lot of params are missing. All the bars which don't span in full length indicate the existence of missing value in it.

<img width="534" alt="Screen Shot 2022-05-22 at 7 08 35 PM" src="https://user-images.githubusercontent.com/25512807/169730384-edf5a130-e339-4773-b384-309cc6391737.png">

#### Figure 4: Visualization of Missing Data using missingo library [10]

 
Once we have dropped values containing ‘-1’, the next step is to look at the missing values. There are three main reasons why values could be missing – Missing at random, Missing Completely at random, Not Missing at random. The initial approach initiated for imputing it using mean imputation. As the name suggests, the mean is calculated for the available values and replaced with the non-missing value’s number. An essential step to bear in mind during mean imputation is to remove outliers to prevent seeing absurd or surprising values as mean.


### A. Mean Imputed Data Analysis:
The data is missing at random in our Dataset, and there is no correlation with other variables  in the dataset. So, the strategy here is to use Mean Imputation to impute the Null or NAN values. 
We used the SimpleImputer class to accomplish this. SimpleImputer is a scikit-learn class for dealing with missing data in a predictive model dataset. 
The mean of the corresponding column is used to replace all missing data.
The data is then shown to see if there are any Null values. It is clear from the Bar Plot below that there are no more Null values.

<img width="496" alt="Screen Shot 2022-05-22 at 7 11 33 PM" src="https://user-images.githubusercontent.com/25512807/169730453-1bca996c-5484-4598-a4f9-08ecc43157b2.png">

#### Figure 5: Visualization after Mean Imputing Data

### B. KNN Imputed Data Analysis
The Nan values or the null data is imputed using KNN imputer with nearest neighbor - 7 and the distance measure used is euclidean distance. 
The stratified split size of test data to train data is 25%.
Data is standardized using StandardScalar() function.
We have used 2 classifiers for analyzing the data : Logistic Regression and Random Forest Classifier.
We divided the data into categorical and numerical columns for better analysis.
Below is the visualization of  categorical in knn imputed data.


<img width="526" alt="Screen Shot 2022-05-22 at 7 12 17 PM" src="https://user-images.githubusercontent.com/25512807/169730517-cadcbbb5-a07a-416b-880e-27abaa2b5727.png">

#### Figure 6: Visualization of categorical data in knn imputed data


We’ve made sure to remove highly correlated features to avoid introducing redundancy in the model.
We’ve made sure to hypertune the parameters to avoid overfitting due to high dimensional data.
And standardize the dataset which is the requirement for most machine learning estimators as they may behave badly if the individual features do not more or less look like standard normally distributed data.
We’ve made sure to optimize the hyperparameters of the models used to use the best parameters and achieve the best accuracy of the model without overfitting it. We have mainly used two techniques, GridSearchCV and RandomSearchCV from sci-kit learn. A key difference is that RandomSearchCV does not test all parameters. Instead, the search is done at random. Since RandomizedSearchCV allowed us to specify the number of parameter values we seek to test, it was used for extracting important features only as we could base our search iterations on our computational resources or the time taken per iteration. This was done because data was high dimensional and basing our search was helpful and helped in saving the resources. And GridSearchCV, before fitting each model because, now we had performed feature engineering and removed outliers, so we decided to check for every possible combination to find the best features, as now the data was not high dimensional and resources were less used.

# Feature Selection and Importance

## Feature Selection and Importance of Mean Imputed Data
1. Methods for computing a score for each model's input characteristic are referred to as "feature importance." The scores describe the "importance" of each feature. 
2. A higher score suggests that the feature will impact the model used to anticipate a particular variable more.
3. Initially, we Split the Mean Imputed data into 70 % for Training and 30% for Testing the Model.
4. The model we used here is  XGBoost Classifier, a trained XGBoost model that calculates feature importance automatically.
5. The importance scores for each feature are then stored in the trained model's feature importances_ member variable.
6. We select the first 15 important features because the Importance of the following features is almost constant, which is seen below in the Plot. 

<img width="296" alt="Screen Shot 2022-05-22 at 7 16 42 PM" src="https://user-images.githubusercontent.com/25512807/169730941-727b1c1e-97c5-48a2-ab20-dbc8bb12e43b.png">

#### Figure 7: Feature Selection and Importance of Mean Imputed Data

## Feature Selection and Importance of KNN Imputed Data 
1. Examining a model's coefficients is the simplest technique to analyze feature importance. It has some influence on the  forecast if the assigned coefficient is a large number. If the coefficient is 0 or that it has no influence on the forecast accuracy. 
2. XGBoost Classifier has built in feature importance. The more significant the feature, the higher the value of the node probability.
3. We find the highest correlated columns or features in the dataset which turns out to be 14 as they provide the same information.
4. Using the XGBClassifier we find the top most important features which determine if the website is a phishing website. The optimal number of features is 34 in our case. 
5. Figure 8 shows the graph for recursive feature elimination using cross validation.\

<img width="301" alt="Screen Shot 2022-05-22 at 7 18 07 PM" src="https://user-images.githubusercontent.com/25512807/169731081-b0bfcacb-189f-4947-9962-a136577404c6.png">

#### Figure 8: Graph for Recursive Feature Elimination using cross validation

6. Here you can observe that after as the number of features increases initially the accuracy also increases but gets saturated once the number of features reaches the number 30.
7. We also plot the feature importance graph which displays the features which affect the result in the most impactful way.
8. Optimal number of features = 42 is obtained using the RFECV method.

<img width="301" alt="Screen Shot 2022-05-22 at 7 18 51 PM" src="https://user-images.githubusercontent.com/25512807/169731157-9891c485-bb9e-42e7-93ec-3f5efd4219d7.png">

#### Figure 9: Feature Selection and Importance of KNN Imputed Data 

## Modeling Comparisons
The accuracy is obtained using various models based on F1 scores bacause we aim to find accuaracy for which true predictions were correct and it generates a single score that accounts for both precision and recall concerns in a single number. The first two models were trained on Mean imputed data while the other models were trained on KNN imputed data. The following are the models:

### Logistic Regression:
Despite doing feature extraction, Hypertuning parameters and regularization, this classifier achieved accuracy found of 99.99 percent via Mean imputed data. Hence, it can be understood that this model obviously overfitting. We predict that it could be some of the features are highly correlated due to mean imputation, as mean imputation can give unpreictable outliers.

### KNN Classifier:
The k-nearest neighbors (KNN) algorithm is a simple, supervised machine learning algorithm that can be used to solve both classification and regression problems. The accuracy found in the KNeighbour classifier is 95.34 percent via Mean imputed data. Below, the figure shows us the accuracy of the KNN Classifier model. We can see in the confusion matrix that the number of False Negatives and False Positives is less than True Positives and True Negatives, which is a good sign. \
![image](https://user-images.githubusercontent.com/90728105/169725911-df1bcd4c-9789-429c-8d2b-3de7ce23928d.png)
#### Figure 10: Accuracy of the KNN Classifier model  

### XGBoost Classifier:
XGBoost is an algorithm that has recently been dominating applied machine learning and Kaggle competitions for structured or tabular data. XGBoost is an implementation of gradient boosted decision trees designed for speed and performance. We can see in the confusion matrix that the number of False Negatives and False Positives is less than True Positives and True Negatives, which is a good sign. The accuracy found through XGB CLassifier is 92 percent via KNN imputation. The performance metrics of XGBoost Classifier can be seen in the figure below- \
![image](https://user-images.githubusercontent.com/90728105/169725626-91195074-df1f-4746-9202-77c6caf3c950.png)
#### Figure 11: Performance Metrics of the KNN Classifier model 

### Logistic Regression:
This time we dopped highly correlated features and then feature extraction to avoid overfitting in KNN imputation.The accuracy found through Logistic Regression is 89.07 percent via KNN imputation. Below, the figure shows us the accuracy of the KNN Classifier model. We can see in the confusion matrix that the number of False Negatives and False Positives is less than True Positives and True Negatives, which is a good sign. Performance metrics of Logistic Regression can be seen in the figure below- \
![image](https://user-images.githubusercontent.com/90728105/169725560-b18fe537-561b-4bd7-b9fc-c586ca641c45.png)
#### Figure 12: Performance Metrics of the Logistic Regression 

### Random Forest Classifier:
When a large number of decision tree operate as an ensemble, they make up Random Forest. Each tree in the random forest produces a class prediction, and the class with the most votes becomes the prediction of our model. The accuracy found through Random Forest is 95 percent via KNN imputation. Below, the figure shows us the accuracy of the KNN Classifier model. We can see in the confusion matrix that the number of False Negatives and False Positives is less than True Positives and True Negatives, which is a good sign. The Performance metrics of Random Forest Classifier can be seen in the figure below- \
![image](https://user-images.githubusercontent.com/90728105/169725726-2876dff8-3258-489d-b5bb-d6ce9e899b71.png)
#### Figure 13: Performance Metrics of the Random Forest Classifier 

## Comparisons
We modeled and analyzed data with 2 imputation methods for which we observed different results because we wanted to demonstrate the difference between Mean ans KNN imputed data analysis and show when the similarity/co-relation among the attributes is considered, unlike Mean imputation. Logistic Regression was applied to both the types of data but it produced an accuracy of 88% on KNN imputed data whereas for mean imputed data, despite regularization and feature extraction it was overfitting.   Models applied to the 2 types of imputed data provided highest accuracy rates of 94%.

## Conclusion
Here, after all the analysis we can conclude that the best accuracy of 94.63% for KNN imputed data was achieved using RandomForestClassifier. The best accuracy for mean imputed data was provided using KNNClassifier which produced an accuracy of 95.34%. And imputation techniques like KNN are better than mean as former preserves corelation among features, whereas latter it may look like there is a stronger relationship than there really is, which isn't good option.
According to feature importance graphs we can conclude the most relevant features correspond to attributes depending on URL and external services, according to both KNN imputed data analysis and mean imputed data analysis.

## References
[1] 6.4. imputation of missing values. scikit. (n.d.). Retrieved May 22, 2022, from https://scikit-learn.org/stable/modules/impute.html 

[2] Brownlee, J. (2019, August 22). A gentle introduction to data visualization methods in Python. Machine Learning Mastery. Retrieved May 22, 2022, from                   https://machinelearningmastery.com/data-visualization-methods-in-python/ 

[3] Brownlee, J. (2020, August 20). How to calculate feature importance with python. Machine Learning Mastery. Retrieved May 22, 2022, from                                 https://machinelearningmastery.com/calculate-feature-importance-with-python/ 

[4] Interquartile range and quartile deviation using NumPy and SciPy. GeeksforGeeks. (2020, June 7). Retrieved May 22, 2022, from                                           https://www.geeksforgeeks.org/interquartile-range-and-quartile-deviation-using-numpy-and-scipy/ 

[5] Lewinson, E. (2021, August 26). Explaining feature importance by example of a random forest. Medium. Retrieved May 22, 2022, from                                       https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e 

[6] McKenzie, C., Morrow, S., Belding, G., Mallory, P., Messina, G., Tavares, P., &amp; Antipov, A. (2022, March 20). Phishing archives. Infosec Resources.                 Retrieved May 22, 2022, from https://resources.infosecinstitute.com/topics/phishing/#gref 

[7] Phishing detection using machine learning techniques - arxiv.org. (n.d.). Retrieved May 23, 2022, from https://arxiv.org/pdf/2009.11116.pdf 

[8] Vrbančič, G. (2020, September 24). Phishing websites dataset. Mendeley Data. Retrieved May 22, 2022, from https://data.mendeley.com/datasets/72ptz43s9v/1 

[9] ResidentMario. (n.d.). Residentmario/Missingno: Missing data visualization module for python. GitHub. Retrieved May 22, 2022, from                                      https://github.com/ResidentMario/missingno 
