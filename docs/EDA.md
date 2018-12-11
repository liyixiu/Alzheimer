---
layout: default
---

[Introduction](./index.html).| [Exploratory data analysis](./EDA.html).| [Models](./Models.html).| [Conclusion](./Conclusion.html).| [References](./References.html).

# Data Description

We use data from ADNI dataset. This dataset is a longitudinal dataset, participants were carried forward from previous phases for continued monitoring while new enrollees were added with each phase to further investigate the evolution of AD. Participants of the study are between 55 to 90 years old, who are from 57 sites in the US and Canada. Participants would go through a series of initial tests and take repeated at intervals over subsequent years. Tests include clinical evaluation, neuropsychological tests, genetic testing, lumbar puncture, and MRI and PET scans. In this dataset, clinical data are obtained on for ADNI protocols, ADNI 1, ADNI 2, ADNI 3, ADNI GO, which comprise clinical information about each subject including recruitment, demographics, physical examinations, and cognitive assessment data, etc. Although they are not all the same regarding the date and completeness of measurement, as we are mainly focused on baseline biological and medical data and screening data for demographic, neurological exams, screening labs, vital signs, etc., which are almost complete for the four protocols, we do not differentiate different ADNI protocols when analyzing the data.

For the use of exploratory data analysis (EDA), we first deleted observations that are not baseline values and observations that the response value ‘DX’ is missing. Since we just included necessary, not repetitive, baseline data and non-patient reported data, we dropped 58 variables inclduing ‘PTID’, ’SITE’, ‘COLPROT’, ‘ORIGPROT’, ‘EXAMDATE’, and all variables do not end with ‘bl’, ‘RID’, ‘Month’, ‘M’, ‘updated_stamp’, ‘FLDSTRNG’, ‘FSVERSION’, and all Ecog test results by the patients. Moreover, considering the completeness of data, we dropped variables that have missing values more than 80%. 

**Fig 1. Completeness of dataset after dropping unnecessary variables and observations**
![percent of not nans for each var](https://github.com/liyixiu/Alzheimer/blob/master/docs/figures/1.jpeg?raw=true)

Next, we identified continuous and categorical variables and converted the later to numbers for the convenience of further manipulation. To explore the current dataset, we first took a look at the correlations between any pairs of the variables without imputing missing values because the loss of data is small when doing correlation analysis, the probability of two variables either has a NaN is small, so we just delete any pair that either variable has a NaN. And the correlation matrix is drawn as below. 

**Fig 2. Correlation matrix including missing values after dropping unnecessary variables and observations**
![Correlation matrix](https://github.com/liyixiu/Alzheimer/blob/master/docs/figures/2.jpeg?raw=true)

