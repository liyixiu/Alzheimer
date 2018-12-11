---
layout: default
---

[Introduction](./index.html).|  [Exploratory data analysis](./EDA.html).|  [Models](./Models.html).|  [Conclusion](./Conclusion.html).|  [References](./References.html).

# Exploratory data analysis

## Data Description

We use data from ADNI dataset. This dataset is a longitudinal dataset, participants were carried forward from previous phases for continued monitoring while new enrollees were added with each phase to further investigate the evolution of AD. Participants of the study are between 55 to 90 years old, who are from 57 sites in the US and Canada. Participants would go through a series of initial tests and take repeated at intervals over subsequent years. Tests include clinical evaluation, neuropsychological tests, genetic testing, lumbar puncture, and MRI and PET scans. In this dataset, clinical data are obtained on for ADNI protocols, ADNI 1, ADNI 2, ADNI 3, ADNI GO, which comprise clinical information about each subject including recruitment, demographics, physical examinations, and cognitive assessment data, etc. Although they are not all the same regarding the date and completeness of measurement, as we are mainly focused on baseline biological and medical data and screening data for demographic, neurological exams, screening labs, vital signs, etc., which are almost complete for the four protocols, we do not differentiate different ADNI protocols when analyzing the data.

## Variables and Observations cleaning
For the use of exploratory data analysis (EDA), we first deleted observations that are not baseline values and observations that the response value ‘DX’ is missing. Since we just included necessary, not repetitive, baseline data and non-patient reported data, we dropped 58 variables inclduing ‘PTID’, ’SITE’, ‘COLPROT’, ‘ORIGPROT’, ‘EXAMDATE’, and all variables do not end with ‘bl’, ‘RID’, ‘Month’, ‘M’, ‘updated_stamp’, ‘FLDSTRNG’, ‘FSVERSION’, and all Ecog test results by the patients. Moreover, considering the completeness of data, we dropped variables that have missing values more than 80%. 


**Fig 1. Completeness of dataset after dropping unnecessary variables and observations**
![fig1](./figures/1.jpeg)

## Correlation Analysis of Variables

Next, we identified continuous and categorical variables and converted the later to numbers for the convenience of further manipulation. To explore the current dataset, we first took a look at the correlations between any pairs of the variables without imputing missing values because the loss of data is small when doing correlation analysis, the probability of two variables either has a NaN is small, so we just delete any pair that either variable has a NaN. And the correlation matrix is drawn as below. 

**Fig 2. Correlation matrix including missing values after dropping unnecessary variables and observations**
![fig2](https://github.com/liyixiu/Alzheimer/blob/master/docs/figures/2.jpeg?raw=true)

## Distribution Exloration for Variables Grouped by Disease Status

Next, we plot the distribution of the variable across non-disease control (CN) participants, MCI participants, and AD participants. Shown in the below plots, we could see that AD, MCI, and CN are not very different across age(AGE), education level (PTEDUCAT), and intracerebral volume (ICV). The levels of Average FDG-PET of angular, temporal, and posterior cingulate
(FDG), Mini-Mental State Examination(MMSE), Hippocampus, WholeBrain, Entorhinal, Fusiform, middle temporal gyrus (MitTemp), etc. are negatively associated with cognitive impairment severity, while the scores of Clinical Dementia Rating-Sum of Boxes (CDRSB), FAQ, Ventricles, etc. are positively associated with cognitive impairment severity.  

**Fig 3. Distribution of continuous variables grouped by disease status**
![fig3](https://github.com/liyixiu/Alzheimer/blob/master/docs/figures/3.jpeg?raw=true)

**Fig 4. Distribution of categorical variables grouped by disease status**
![fig3](https://github.com/liyixiu/Alzheimer/blob/master/docs/figures/4.jpeg?raw=true)

According to the bar charts, the three health status have different distribution on demographic features. AD has a higher distribution in male, non-Hispanic, especially white, married people, and with APOE4=1. However, we are not sure about the determinants of classification yet since the results may due to the sampling method.

