# Credit Risks: Using Pipelines To Map Default Probability 

Aim:
To analyze the correlations and to predict the loan risk that consumers undertake.

Background:
1. Simply put, Credit is the act of borrowing money for a stipulated period of time which is then repaid with interest.
2. The failure to repay the money borrowed is also known as defaulting.
3. In this dataset, we have 1000 instances of consumers that have taken loans for various purposes.
4. This dataset contains several attributes, and we aim to predict the default rate of the consumer based on these correlations.

Problem Definition:
We aim to predict risks for the modern consumer based on behavioral parameters outlined within the dataset.

Societal Impact:
To intercept and refinance consumers before they default on loans due to poor fiscal health.

We have obtained this dataset from the UC Irvine Machine Learning Repository, and this dataset was developed by Dr. Hans Hofmann at the University of Hamburg. The main parameters have been listed below.
1. Age/Gender/Job
2. Housing 
3. Saving/Checking account
4. Credit amount/Duration/Purpose (in DM)
5. Risk (Good or Bad Risk)

We intend to predict the credit risk of consumers using pipelines.
Pipelines are a linear sequence of data transformations that are chained together, therefore culminating in a modeling process that can be evaluated and fine tuned accordingly.

Concepts Used:
1. Data Preprocessing
2. Financial Concepts of Credit/Debt
3. Classification
4. Encoding
5. Boosting
6. Cross Validation
7. Pipelines
8. Model Selection

While utilizing data from the dataset, we couldn’t fit the model on the training data and couldn’t say that the model would work accurately for the test data.
For this, we must ensure that our model develops the correct patterns from the data, and it is not processing too much noise. 
Hence, we use the cross-validation technique in which we train our model using the subset of the compiled dataset and then evaluate using the complementary portion.
Following the below steps for Cross-Validation:
1. Reserve a portion of the subset.
2. Use the rest of the dataset to train the model.
3. Test the model using a reserved portion of the compiled dataset.
4. Iterate accordingly.

We aim to provide the best compilation of scores based on the cross-validation scores based on the following algorithms, [LR: Logistic Regression, LDA: Linear Discriminant Analysis, KNN: K-Nearest Neighbors, CART: Decision Trees (Classification and Regression Trees), NB: Gaussian Naive Bayes, RF: Random Forest, SVM: Support Vector Machine, XGB: XGBoost (Gradient Boosted Decision Trees)]


Question 1: Why should we use XGBoost?
1. Sparse Aware: Automatic handling of missing data values present in attributes “Checking account” and “Saving accounts”.
2. Block Structure: This supports the parallelization of Decision Tree construction.
3. Continued Training: We will further fine tune our model that has already been fitted model on existing training data.

XGBoost is an extremely advanced boosting algorithm capable of creating new models capable of correcting the errors made by existing models, and as we are using multiple algorithms it is highly likely that XGBoost will correct some of the errors that have crept in during classification procedures.

Question 2: Why Should We Use GNB?
1. We see that our data tends to form continuous values distributed over a slightly normalized Gaussian curve.
2. Hence, we can implement a Gaussian Naïve-Bayes algorithm upon our data as a form of classification.

Question 3: Which Algorithms Cannot Be Used?
1. SGD Regression,
2. Bayesian Ridge, 
3. Lasso Lars,  
4. ARD Regression
5. Passive Aggressive Regressor
6. Theil Sen Regression
7. Linear Regression

We cannot use the algorithms listed above as classification metrics cannot handle a mix of binary and continuous targets. In our project, we have unsuccessfully attempted to to use linear regression and then round/threshold the outputs, effectively treating the predictions as "probabilities" and thus converting the model into a classifier. 
However, while doing so, we received negative errors which adversely affected our model. 

References:
1. Byanjankar, Ajay, Markku Heikkilä, and Jozsef Mezei. "Predicting credit risk in peer-to-peer lending: A neural network approach." 2015 IEEE Symposium Series on Computational Intelligence. IEEE, 2015.
2. Zhu, You, et al. "Predicting China’s SME credit risk in supply chain financing by logistic regression, artificial neural network and hybrid models." Sustainability 8.5 (2016): 433.
3. Khemakhem, Sihem, and Younes Boujelbene. "Predicting credit risk on the basis of financial and non-financial variables and data mining." Review of Accounting and Finance (2018).
4. Yang, Ke, et al. "Fairness-Aware Instrumentation of Preprocessing~ Pipelines for Machine Learning." Workshop on Human-In-the-Loop Data Analytics (HILDA'20). 2020.
5. kaggle.com

