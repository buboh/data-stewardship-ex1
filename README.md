# DS Experiment Description
Moritz Leidinger, 11722966
## 1. Task
This experiment aims to apply two machine learning models to solve a classification task for two different datasets. Each dataset is evaluated with each of the machine learning models, using different parameter settings and preprocessing strategies to compare the respective results and analyze across datasets and/or machine learning models.

## 2. Chosen datasets
Two datasets were used for the experiments; one from Kaggle, the other one from the UCI Machine Learning repository. They are:
 
### Dataset “Breast Cancer” (Kaggle [[1]](https://www.kaggle.com/c/184702-tu-ml-ws-18-breast-cancer/data))
- \# of samples:	285
- \# of features:	32
- \# of classes:	2
- type of data:	    numeric
- missing values:	no
- file format	    .csv (3 files)

### Dataset “Arrhythmia” (UCI [[2]](https://archive.ics.uci.edu/ml/datasets/Arrhythmia))
- \# of samples:	452
- \# of features:	279
- \# of classes:	16
- type of data:	    numeric & categoric
- missing values:	yes
- file format	    .txt (2 files) 

For the Breast Cancer dataset, the task was to predict if a patient has a "recurrence-events" or not ("no-recurrence-events")” . 

The Arrhythmia dataset is described as having 279 attributes, 206 of which are linear valued, the rest are nominal, with their values represented by number codes. The dataset contains no textual data, with the exception of six columns, whose missing values are represented by ‘?’. The classification goal is to “distinguish between the presence and absence of cardiac arrhythmia and classify it in one of the 16 groups. Class 01 refers to 'normal' ECG, classes 02 to 15 refer to different classes of arrhythmia and class 16 refers to the rest of unclassified ones.” At closer inspection it was found, that 3 of the classes (11, 12, 13) are unused in the given dataset, while four classes contain less than six samples each (7, 8, 14, 15). 

## 3. Experiment setup
Jupyter notebooks was used as a programming platform. Implementations of the chosen machine learning models were taken from the scikit-learn [[3]](https://scikit-learn.org/stable/) framework for Python [[4]](https://www.python.org/). The two models chosen were k-Nearest-Neighbours (kNN) and Decision Trees (DTREE).

### 3.1. Preprocessing
The dataset “Breast Cancer” contains good quality numeric data (no missing values or wrong data types etc.), hence, preprocessing was limited scaling of the data using a standardized model [[5]](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) and reducing the number of features using PCA [[6]](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html). Results improved with scaled data, PCA on the other hand, had little effect on the dataset.

To deal with the missing values in the Arrhythmia dataset, the ‘?’-string was first converted to a np.nan, then different combinations of strategies were applied, including dropping all columns containing NaNs, filling NaNs with dummy values (0, -999), filling NaNs with the column mean value or standardizing the data by removing the mean and scaling to unit variance. In order to reduce classification runtime, principal component analysis (PCA) was applied in some cases. When applied, the number of columns was reduced from 279 to sqrt(279) which equals 16 after rounding.

## 4. Discussion
Looking at the overall performance and predictions of the conducted experiments, it was observed that the differences in the dataset size and quality as well as the preprocessing steps had the biggest impact. The best results were achieved with the Breast Cancer dataset who had no missing values and only numeric data. The dataset with mixed and missing values needed more complex preprocessing strategies as the machine learning models were not able to deal with categorical data and no good results were achived without filling in gaps with dummy values. To further push the accuracy of the prediction results, all values (represented as numeric data) were scaled, which also increased the runtime for some of the training algorithms. 

The Arrhythmia dataset achieved quite poor results across all machine learning models since only few classes represented the majority of labels. Some classes were not contained in the training set at all, which lead to a learning model which did not predict any of those classes either.

