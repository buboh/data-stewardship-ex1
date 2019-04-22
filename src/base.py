import pydotplus
import graphviz
import datetime as dt
import pandas as pd
import numpy as np
import sklearn as sk
from pygments import StringIO
from sklearn import neighbors
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# mlpath = "C:/Users/Moritz/Desktop/DS-UE1/"
mlpath = '../'

# Classifiers
# Preprocessing


# scale data
def scale_data(train_data, test_data=pd.DataFrame):
    scaler = preprocessing.StandardScaler()

    # Fit on training set only.
    scaler.fit(train_data)

    # Apply transform to both the training set and the test set.
    train_data[train_data.columns] = pd.DataFrame(scaler.transform(train_data[train_data.columns]))
    if test_data.empty:
        return train_data
    else:
        test_data[test_data.columns] = pd.DataFrame(scaler.transform(test_data[test_data.columns]))
        return train_data, test_data


# strip whitespaces
def strip(data):
    return data.apply(lambda x: x.str.strip())


# replace empty strings with nan
def fillspace_nan(data):
    return data.apply(lambda x: x.replace('', np.nan))


# one hot encoding
def one_hot(data, drop_first=True):
    columns = data.select_dtypes(['object'])
    return pd.get_dummies(data, columns=columns, drop_first=True)


# primary component analysis
def pca(train_data, n_comp):
    pca = PCA(n_components=n_comp)
    pca.fit(train_data)
    pca_train = pd.DataFrame(pca.transform(train_data))
    return pca_train


# Cross-validation
def run_cv(classifier, train_data, train_target, num_cv=10):
    scores = model_selection.cross_validate(classifier, train_data, train_target, cv=num_cv,
                                            scoring=['accuracy', 'f1_macro'])
    print("Accuracy: {:.3f} (+/- {:.3f}), F1: {:.3f} (+/- {:.3f})".format(scores['test_accuracy'].mean(),
                                                                          scores['test_accuracy'].std(),
                                                                          scores['test_f1_macro'].mean(),
                                                                          scores['test_f1_macro'].std()))


# Run KNN
def run_knn(train_data, train_target, test_data, test_target, k=5, col_name='predict', skip_cv=False):
    # define classifier
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    # train
    knn.fit(train_data, train_target)
    # predict
    knnresult = pd.DataFrame(knn.predict(test_data), columns=[col_name])
    # cross validation
    if not skip_cv:
        print("CV: {}".format(k))
        run_cv(knn, train_data, train_target)
    return knnresult, knn


# Create KNN result
def create_knn_result(train_data, train_target, test_data, test_target, k, col_name, skip_cv=False):
    (knnresult, knn) = run_knn(train_data, train_target, test_data, test_target, k, col_name, skip_cv)

    print("KNN", knn.score(test_data, test_target))
    print(metrics.confusion_matrix(test_target, knnresult))

    return knnresult, knn


# Create KNN filename
def knn_filename(k, scale):
    return f"knn_{k}{'_scaled' if scale else ''}_{str(dt.datetime.now())}.csv"


# Run DTREE
def run_dtree(train_data, train_target, test_data, criterion='entropy', max_depth=None, post_prune=False,
              col_name='predict', skip_cv=False, num_cv=10):
    # define classifier
    dtree_clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    # train
    dtree_clf.fit(train_data, train_target)
    # predict
    dtreeresult = pd.DataFrame(dtree_clf.predict(test_data), columns=[col_name])
    # cross validation
    if not skip_cv:
        print("CV: [Criterion] {}, [Max Tree Depth] {}".format(criterion, max_depth))
        run_cv(dtree_clf, train_data, train_target, num_cv)
    return dtreeresult, dtree_clf


# Run DTREE
def run_dtree2(train_data, train_target, test_data, criterion='entropy', max_depth=None, post_prune=False,
               col_name='predict', skip_cv=False, num_cv=10):
    # define classifier
    dtree_clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    # train
    dtree_clf.fit(train_data, train_target)
    # predict
    # dtreeresult = pd.DataFrame(dtree_clf.predict(test_data), columns=[col_name])
    # cross validation
    if not skip_cv:
        print("CV: [Criterion] {}, [Max Tree Depth] {}".format(criterion, max_depth))
        run_cv(dtree_clf, train_data, train_target, num_cv)
    return dtree_clf


# Plot Decision Tree
def plot_tree(dtree_clf):
    dot_data = StringIO()
    export_graphviz(dtree_clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return Image(graph.create_png())


# Create DTREE results
def create_dtree_results(train_data, train_target, test_data, test_target, criterion, max_depth, post_prune, col_name,
                         skip_cv):
    (dtreeresult, dtree_clf) = run_dtree(train_data, train_target, test_data, criterion, max_depth,
                                         post_prune, col_name, skip_cv)

    print("DTREE ", dtree_clf.score(test_data, test_target))
    print(metrics.confusion_matrix(test_target, dtreeresult))
    return dtreeresult, dtree_clf


# Create DTREE filename
def dtree_filename(criterion, max_depth, post_prune):
    return 'dtree_{}_{}{}_{}.csv'.format(criterion, str(max_depth), '_pp' if post_prune else '', str(dt.datetime.now()))
