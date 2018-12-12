---
layout: page
title: "Models"
description: "Model fitting and tuning based on python"
header-img: "img/home-bg.jpg"
---

# Contents
1. Disease (MCI and AD) vs. Non-diseases
1.1 Baseline (Simple logistic regression model)
1.2 Optimised logistic regression model
1.3 Random forest
1.4 Dicision tree
1.5 Boosting
1.6 Quadratic discriminant analysis (QDA)
2. MCI vs. AD
2.1 Baseline (Simple logistic regression model)
2.2 Optimised logistic regression model
2.3 Random forest
2.4 Dicision tree
2.5 Boosting
2.6 Quadratic discriminant analysis (QDA)

## 1. Disease (MCI and AD) vs. Non-diseases

On the basis of a criterion for the quality of life and intervention decision, distinguishing disease from non-disease is of great importance. Thus, we provided 6 ways to conduct the classification including logistic regression, random forest, decision tree, boosting, QDA, and random forest models.

### 1.1 Baseline (Simple logistic regression model)
We fitted the logistic regression model with split train and test dataset. With all covariates, we got an accuracy score of 0.733 for the training set and 0.723 for the test set.

```python
#LogisticRegression with all covariates
ovr=LogisticRegressionCV(multi_class = 'ovr', cv=5)
ovr_model=LogisticRegressionCV(multi_class = 'ovr', cv=5).fit(X_train1,y_train1)
print('The accuracy in training dataset is '+"{}".format(ovr_model.score(X_train1, y_train1)))
print('The accuracy in testing dataset is '+"{}".format(ovr_model.score(X_test1, y_test1)))
```
The accuracy in training dataset is 0.7328621908127209\\
The accuracy in testing dataset is 0.7231638418079096\\

### 1.2 Optimised logistic regression model

We conducted forward selection to select the most valuable predictors. As the graph has shown, the graph described the accuracy of the models that we get through forward selection. At each point, the model’s accuracy was drawn from variables from the beginning until that point in the test set. The graph described the accuracy of the models that we get through forward selection. From the output, we chose only CDRSB as our predictor in the model as it will generate an accuracy score of 0.978 for training set and 0.963 for test set.

```python
# forward selection for logistic model
# input 'model' is a pre-fit model like LogisticRegressionCV(multi_class = 'ovr', cv = 5)
def forward_selection(model, X_train, X_test, y_train, y_test):
    var_selected = []
    accuracy = []
    for i in range(len(list(X_train))):
        X_train_rest = X_train.drop(var_selected, axis = 1)
        accuracy_temp = []
        for j in range(len(list(X_train_rest))):
            var_now = var_selected.copy()
            var_now.append(list(X_train_rest)[j])
            model_fitted = model.fit(X_train[var_now], y_train)
            y_test_pred = model_fitted.predict(X_test[var_now])
            accuracy_temp.append(accuracy_score(y_test, y_test_pred))
        var_selected.append(list(X_train_rest)[accuracy_temp.index(max(accuracy_temp))])
        accuracy.append(max(accuracy_temp))
    return(var_selected, accuracy)
```
```python
model = LogisticRegressionCV(multi_class = 'ovr', cv = 5)
var_selected1, accuracy1 = forward_selection(model, X_train1, X_test1, y_train1, y_test1)
plt.figure(figsize=(15,5))
plt.plot(var_selected, accuracy)
plt.xticks(rotation=90)
plt.ylabel('accuracy on testing dataset')
plt.title('Accuracy for forward selection on prediction of controls and diseases')
plt.show()
```
**Fig 5. forward selection for logistic regression**
<img src="https://yueli1201.github.io/Alzheimer/figures/5.jpeg" alt="6" width="750"/>

```python
## best logistic
X_train_restrict=np.array(X_train1['CDRSB']).reshape(-1,1)
X_test_restrict=np.array(X_test1['CDRSB']).reshape(-1,1)
ovr_model_sparse=LogisticRegressionCV(multi_class = 'ovr', cv=5).fit(X_train_restrict, y_train1)
print('The accuracy in training dataset is '+"{}".format(ovr_model_sparse.score(X_train_restrict, y_train1)))
print('The accuracy in testing dataset is '+"{}".format(ovr_model_sparse.score(X_test_restrict, y_test1)))
```
The accuracy in training dataset is 0.9780918727915194\\
The accuracy in testing dataset is 0.963276836158192\\

### 1.3 Random forest

Without tuning the parameters, random forest generates an accuracy score of 0.958 and AUC of 0.972, which indicated good prediction and classification capability. After tuning the parameters, we chose depth = 50 and number of estimaters = 100 and got an accuracy score of 0.960 and AUC of 0.984.

```python
###Random forest baseline model
rf1 = RandomForestClassifier(random_state=100)
display(rf1.fit(X_train1, y_train1))
```
```python
##Tuning parameters for random forest
depths = [1,5,10,50]
num = [10, 20, 50,100]
tune = pd.DataFrame(np.zeros((len(depths)*len(num), 3)))

i = 0
kf = KFold(n_splits=5)

for d in depths:
    for n in num:
        print("depth:{}, n:{}".format(d,n))
        tune_cv = []
        for train, val in kf.split(X_train1):
            train_X, train_y, val_X, val_y = X_train1.iloc[train,:], pd.DataFrame(y_train1).iloc[train,:], X_train1.iloc[val,:], pd.DataFrame(y_train1).iloc[val,:]
            rf_model1 = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=100)
            rf_model1.fit(train_X, train_y.values.ravel())
            tune_cv.append(roc_auc_score(val_y, rf_model1.predict_proba(val_X)[:,1]))

        tune.iloc[i,:] = [d, n, np.mean(tune_cv)]
        i+=1
        
tune.columns = ['max_depth','max_num','AUC']
tune.sort_values(by = 'AUC', ascending=False)
```
```python
# choose the best rf model based on above
rf1_best = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=100)
rf1_best_model=rf1_best.fit(X_train1, y_train1)

y_pred_rf1 = rf1.predict(X_test1)
y_pred_rf1_best = rf1_best_model.predict(X_test1)
accuracy_rf1 = accuracy_score(y_pred_rf1, y_test1)
accuracy_rf1_best = accuracy_score(y_pred_rf1_best, y_test1)
print('Accuracy score for random random forest model is', accuracy_rf1)
print('Accuracy score for best random forest model is', accuracy_rf1_best)
```
Accuracy score for random random forest model is 0.9576271186440678\\
Accuracy score for best random forest model is 0.96045197740113\\

```python
### ROC for baseline random forest
fpr11, tpr11, thres11 = roc_curve(y_test1, rf1.predict_proba(X_test1)[:,1])
### ROC for bset random forest
fpr11_best, tpr11_best, thres11_best = roc_curve(y_test1, rf1_best.predict_proba(X_test1)[:,1])
fpr12_best, tpr12_best, thres12_best = roc_curve(y_test1, np.zeros((len(y_test1), 1)))
auc11_best = roc_auc_score(y_test1, rf1_best.predict_proba(X_test1)[:,1])
auc12_best = roc_auc_score(y_test1, np.zeros((len(y_test1), 1)))

fig, ax = plt.subplots(1,1)
ax.plot(fpr11, tpr11, '-', alpha=0.8, label='Baseline random forest' % (auc11))
ax.plot(fpr11_best, tpr11_best, '-', alpha=0.8, label='Best random forest' % (auc11_best))
ax.plot(fpr12_best, tpr12_best, '-', alpha=0.8, label=' 0 classifier' % (auc12_best))
plt.title("ROC curves for random forest model (Disaese vs. Non-disease)")
plt.xlabel('1-specificity')
plt.ylabel('sensitivity')
plt.legend()
plt.show()
```

**Fig 6. ROC curves for random forest model (Disaese vs. Non-disease)**
<img src="https://yueli1201.github.io/Alzheimer/figures/6.tiff" alt="6" width="750"/>

On the basis of the selected model, we counted the times that each feature was used at the top node and concluded EcogSPMem, FAQ, and CDRSB, which are a bit different from the output of forward selection generated by regression analysis.

```python
y_train1=pd.DataFrame(y_train1)
y_train1['tmp'] = 1
X_train1['tmp'] = 1
y_train1.columns=['DX','tmp']
train1 = pd.merge(X_train1, y_train1, on=['tmp'])
train1 = train1.drop('tmp', axis=1)
X_train1 = X_train1.drop('tmp',axis=1)
y_train1 = y_train1.drop('tmp',axis=1)

columns=X_train1.columns
#random forest
nrf = [0]*X_train1.shape[1]
for i in range(50):
    for j in range(X_train.shape[1]):
        if rf1_best.estimators_[i].tree_.feature[0] == j:
            nrf[j] = nrf[j]+1

plt.figure(figsize = (20,6))
plt.bar(columns, nrf, width = 0.4, label = 'random forest')
plt.xlabel('$features$',fontsize=16)
plt.ylabel('$counts$',fontsize=16)
plt.title('Times that each feature is used at the top node',fontsize=16)
plt.legend(loc = 'best',fontsize=16)
plt.xticks(rotation=90)
plt.show()
```

**Fig 7. Times that each feature is used at the top node**
<img src="https://yueli1201.github.io/Alzheimer/figures/7.jpeg" alt="7" width="750"/>

### 1.4 Dicision tree

