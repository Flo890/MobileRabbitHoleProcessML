from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import export_text
from sklearn import metrics
import ML_helpers

dataframe_dir_users = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\users'

df_sessions = pd.read_pickle(fr'{dataframe_dir_users}\user-sessions_features_labeled.pickle')

def svm_classifier(x, y):
    print("***SVM***")
    # save the feature name and target variables
    feature_names = x.columns
    labels = y.unique()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # svclassifier = SVC(kernel='poly', degree=8) # -> very long
    # svclassifier = SVC(kernel='linear') -> very long

    # svclassifier = SVC(kernel='rbf')
    svclassifier = SVC(kernel='sigmoid')
    svclassifier.fit(x_train, y_train)

    y_predict = svclassifier.predict(x_test)

    print("-----score----------")
    score = metrics.accuracy_score(y_test, y_predict)
    print(score)
    print("-----report----------")
    # print(metrics.confusion_matrix(y_test,y_predict))
    print(metrics.classification_report(y_test,y_predict))



def kNeighbors_classifier(x, y):
    print('***KNeighborsClassifier***')

    labels = y.unique()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    KNN = KNeighborsClassifier(n_neighbors=3)
    KNN.fit(x_train, y_train)

    #Predict Output
    y_predict = KNN.predict(x_test)

    print("-----score----------")
    score = metrics.accuracy_score(y_test, y_predict)
    print(score)
    print("-----report----------")
    print(metrics.classification_report(y_test, y_predict))


def random_forest_classifier(x, y):
    print('***Random Forest***')

    feature_list = x.columns # list(x_features.columns)
    x = np.array(x)
    labels = df_sessions['target_label'].unique()
    y = np.array(y)

    x_train_features, x_test_features, y_train_labels, y_test_labels = train_test_split(x, y, test_size=0.25, random_state=42)
    # print('Training Features Shape:', x_train_features.shape)
    # print('Training Labels Shape:', y_train_labels.shape)
    # print('Testing Features Shape:', x_test_features.shape)
    # print('Testing Labels Shape:', y_test_labels.shape)

    forest = RandomForestClassifier(criterion='gini', n_estimators=5, random_state=42, n_jobs=2)
    forest.fit(x_train_features, y_train_labels)

    y_predict = forest.predict(x_test_features)

    print("-----score----------")
    score = metrics.accuracy_score(y_test_labels, y_predict)
    print(score)

    print("-----report----------")
    print(metrics.classification_report(y_test_labels, y_predict))

    print("-------importance--------")
    importance = pd.DataFrame({'feature': feature_list, 'importance': np.round(forest.feature_importances_, 3)})
    importance.sort_values('importance', ascending=False, inplace=True)
    print(importance)


    # print('--------Precision-------')
    # precision = metrics.precision_score(y_test_labels, y_predict, average=None)
    # precision_results = pd.DataFrame(precision, index=labels)
    # precision_results.rename(columns={0: 'precision'}, inplace=True)
    # print(precision_results)


def decision_tree_classifier(x, y):
    print('***Decision tree***')
    feature_names = x.columns
    labels = y.unique()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    clf_model = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42, min_samples_leaf=5)
    clf_model.fit(x_train, y_train)

    # tree_rules = export_text(clf_model, feature_names=list(feature_names))
    # print(tree_rules)

    y_predict = clf_model.predict(x_test)

    print("-----score----------")
    score = metrics.accuracy_score(y_test, y_predict)
    print(score)
    print("-----report----------")
    print(metrics.classification_report(y_test, y_predict))

    print("-------importance--------")
    importance = pd.DataFrame({'feature': x_train.columns, 'importance': np.round(clf_model.feature_importances_, 3)})
    importance.sort_values('importance', ascending=False, inplace=True)
    print(importance)

    # print('--------Precision-------')
    # precision = metrics.precision_score(y_test, y_predict, average=None)
    # precision_results = pd.DataFrame(precision, index=labels)
    # precision_results.rename(columns={0: 'precision'}, inplace=True)
    # print(precision_results)
    #
    # print("-------Recall-------------")
    # recall = metrics.recall_score(y_test, y_predict, average=None)
    # recall_results = pd.DataFrame(recall, index=labels)
    # recall_results.rename(columns={0: 'Recall'}, inplace=True)
    # print(recall_results)

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    df_sessions = pd.read_pickle(fr'{dataframe_dir_users}\user-sessions_features_labeled.pickle')
    x, y = ML_helpers.prepare_df(df_sessions)

    decision_tree_classifier(x, y)
    random_forest_classifier(x, y)
    kNeighbors_classifier(x, y)

    # x, y = prepare_df_no_oversampling(df_sessions)
    # svm_classifier(x, y)

