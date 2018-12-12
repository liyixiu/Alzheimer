---
layout: page
title: "Results & Conclusion"
description: "???"
header-img: "img/home-bg.jpg"
---

# Results

## Comparation of models for disease (MCI and AD) vs. non-disease

To compare the performance of reproducibility and classification for the above models, we printed an accuracy table and drew a ROC curve. 
The accuracy table showed except for the baseline logistic regression model, the other models all generated excellent accuracy score for both training and test dataset. Scores for the training set are always higher than the test set.

```python
## comparasion of accuracy of different model
classifiers=[ovr_model, rf1_best_model,   best_tree_model, AdaBoost_model, qda_model]
names = ['baseline logistic','random forest', 'decision tree', 'boosting','qda']
accuracy_table=pd.DataFrame()
j=0
for i in classifiers:
    accuracy_table.loc[names[j],r'$ accuracy(train)$']=accuracy_score(y_train1, i.predict(X_train1))
    accuracy_table.loc[names[j],r'$ accuracy(test)$']=accuracy_score(y_test1, i.predict(X_test1))
    j=j+1

accuracy_table.loc['optimal logistic',r'$ accuracy(train)$']=accuracy_score(y_train1, ovr_model_sparse.predict(X_train_restrict))
accuracy_table.loc['optimal logistic',r'$ accuracy(test)$']=accuracy_score(y_test1, ovr_model_sparse.predict(X_test_restrict))
accuracy_table
```

**Table 1. Accuracy table (Disease vs. Non-disease)**
<img src="https://yueli1201.github.io/Alzheimer/figures/t1.png" alt="t1" width="750"/>

However, accuracy score alone just told us the predictive capability of models, thus ROC curve and AUC calculations were further explored to provide the information of classification. While the baseline logistic model gave the lowest classification performance, the other models all have pretty good AUC score. We will choose random forest and boosting as our final models as they have excellent performance in both reproducibility and classification.

```python
### ROC for decision tree
fpr_dt11, tpr_dt11, thres_dt11 = roc_curve(y_test1, best_tree.predict_proba(X_test1)[:,1])
fpr_dt12, tpr_dt12, thres_dt12 = roc_curve(y_test1, np.zeros((len(y_test1), 1)))
auc_dt11 = roc_auc_score(y_test1, best_tree.predict_proba(X_test1)[:,1])
auc_dt12 = roc_auc_score(y_test1, np.zeros((len(y_test1), 1)))
### ROC for bset random forest
fpr_qda11, tpr_qda11, thres_qda11 = roc_curve(y_test1, qda_model.predict_proba(X_test1)[:,1])
auc_qda11 = roc_auc_score(y_test1, qda_model.predict_proba(X_test1)[:,1])
###ROC for boosting
fpr_boo11, tpr_boo11, thres_boo11 = roc_curve(y_test1, AdaBoost_model.predict_proba(X_test1)[:,1])
auc_boo11 = roc_auc_score(y_test1, AdaBoost_model.predict_proba(X_test1)[:,1])
###ROC for random forest
fpr11_best, tpr11_best, thres11_best = roc_curve(y_test1, rf1_best.predict_proba(X_test1)[:,1])
auc11_best = roc_auc_score(y_test1, rf1_best.predict_proba(X_test1)[:,1])
###ROC for logistic regression
fpr_ovr11, tpr_ovr11, thres_ovr11 = roc_curve(y_test1, ovr_model_sparse.predict_proba(X_test_restrict)[:,1])
auc_ovr11 = roc_auc_score(y_test1, ovr_model_sparse.predict_proba(X_test_restrict)[:,1])

fig, ax = plt.subplots(1,1)
ax.plot(fpr_dt11, tpr_dt11, '-', alpha=0.8, label='Best decision tree' % (auc_dt11 ))
ax.plot(fpr_qda11, tpr_qda11, '-', alpha=0.8, label='QDA' % (auc_qda11 ))
ax.plot(fpr_boo11, tpr_boo11, '-', alpha=0.8, label='Boosting' % (auc_boo11 ))
ax.plot(fpr11_best, tpr11_best, '-', alpha=0.8, label='Best random forest' % (auc11_best))
ax.plot(fpr_ovr11, tpr_ovr11, '-', alpha=0.8, label='Optimized egression' % (auc_ovr11))
ax.plot(fpr_dt12, tpr_dt12, '-', alpha=0.8, label=' 0 classifier' % (auc_dt12))
plt.title("ROC curves model for disease and non-disease")
plt.xlabel('1-specificity')
plt.ylabel('sensitivity')
plt.legend()
plt.show()
```
```python
## comparison of accuracy of different model
classifiers=[ovr_model, rf1_best_model, best_tree_model, AdaBoost_model, qda_model]
names = ['baseline logistic','random forest', 'decision tree', 'boosting','qda']
AUC_table=pd.DataFrame()
j=0
for i in classifiers:
    AUC_table.loc[names[j],r'$ AUC $']=roc_auc_score(y_test1, i.predict_proba(X_test1)[:,1])
    j=j+1

AUC_table.loc['optimized logistic',r'$ AUC $']=accuracy_score(y_test1, ovr_model_sparse.predict(X_test_restrict))
AUC_table
```

**Fig 15. ROC for different models for disease vs. non-disease**
<img src="https://yueli1201.github.io/Alzheimer/figures/15.png" alt="15" width="750"/>

## Comparation of models for AD vs. MCI vs. non-disease

Likewise, we printed an accuracy table and a ROC curve to compare the performance of reproducibility and classification for the above models. Similar to the previous comparison, baseline logistic gave the lowest accuracy score, while random forest, boosting still have the highest performance in terms of reproducibility.

```python

```


# Conclusion

# Future Work

