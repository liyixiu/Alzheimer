---
layout: page
title: "About"
description: "Why use Project Pages?"
header-img: "img/home-bg.jpg"
---

[Introduction](./index.html).|  [Exploratory data analysis](./EDA.html).|  [Models](./Models.html).|  [Conclusion](./Conclusion.html).|  [References](./References.html).

# Models

## Covariates Selection and Imputation of Missing Data

We have a longitudinal dataset. But for our classification analysis, we just want to keep the baseline information to build classification models. Because, first, we want to find a cost-efficient way to help the classification of AD. Secondly, in the longitudinal data, the information is highly correlated within each individual. 'Examdate', 'update_stamp', 'FLDSTRENG', 'FSVERSION' are  are not useful for the mdoel because they are not the relavent information of patients.  So, we excluded them from analysis. According many previous publications (ref), the patient everyday cognition scale (Ecog) is very uninformative, especially among those dementia people. So all EcogPt variables were excluded from our analysis. Only very few participants in ADNI1 (less than 5% of the total data) have information on Pittsburgh compound B (PIB) test. Therefore, this variable was excluded from our analysis. 

Among the rest of the data, Participants in ADNI1 don’t have information on Everyday Cognition Scale (Ecog), Montreal Cognitive Assessment (MOCA), and AV45. Participants from ADNI3 lack information on APOE4, FDG-PET, AV45, Hippocampus volume, whole brain status, Entorhinal, Fusiform, middle temporal gyrus (MidTemp), intracerebral volume (ICV), Ventricles. But as there are only 46 participants in ADNI3, it will not cause a large proportion missing. There are some other randomly missing data. When combine participants recruited based on four different protocols, we assumed that the data were missing at random.  We used IterativeImputer method in fancyimpute (A strategy for imputing missing values by modeling each feature with missing values as a function of other features in a round-robin fashion) to impute these missing data. 

```python
# Impute missing data
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler

X=df6.drop(['DX'], axis=1)
Y=df6['DX']
columns = X.columns

X_filled_ii = IterativeImputer().fit_transform(X)
X_filled_ii = pd.DataFrame(X_filled_ii)
X_filled_ii.columns = X.columns
```

```python
# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(X_filled_ii, Y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
```
(1415, 33) (1415,)\\
(354, 33) (354,)

## Baseline Model

In order to find the optimal approach for the classification of AD, MCI, and controls, we compare several baseline models using different algorithms with random forest, decision tree, simple and polynomial logistic regression,  boosting, LDA and QDA. 

Covariates are selected based on expert knowledge and previous publications. Demographic variables (race, ethnicity, age, years of education, marriage status, etc.) and biological information (FDG, CDR-SB, MMSE, ventricles volume, hippocampus volume, whole brain volume, etc.) were included as predictors. All cognitive tests are excluded because these tests are similar assessments as AD diagnosis. 

To compare the performance of the above classification methods, we output the accuracy scores for both training set and test set using all the different models. Decision tree method yields the highest accuracy score, and random forest ranked the second highest. Since the decision tree method may yield high bias when the depth=3 (the highest accuracy of the decision tree method), we would like to choose random forest as our final model for classification. 

### Logistic Regression Model

```python
# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
ovr = LogisticRegressionCV(multi_class = 'ovr', cv = 5)
ovr_model = LogisticRegressionCV(multi_class = 'ovr', cv = 5).fit(X_train,y_train)
polynomial_logreg_estimator = make_pipeline(
    PolynomialFeatures(degree = 2, include_bias = False),
    LogisticRegressionCV(multi_class = "ovr", cv = 5))
poly_model = polynomial_logreg_estimator.fit(X_train, y_train)
print('Accuracy of logistic regression model on train set is', np.mean(cross_val_score(ovr_model, X_train, y_train, cv = 5)))
y_pred_test = ovr_model.predict(X_test)
print('Accuracy of logistic regression model on test set is', accuracy_score(y_test, y_pred_test))
```
Accuracy of logistic regression model on train set is 0.5766787099561508\\
Accuracy of logistic regression model on test set is 0.5790960451977402

