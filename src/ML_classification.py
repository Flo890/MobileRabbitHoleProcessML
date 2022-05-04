from collections import Counter

import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import ML_helpers
from imblearn.ensemble import BalancedRandomForestClassifier

dataframe_dir_ml = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\ML'
dataframe_dir_ml_labeled = f'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\ML\labled'
dataframe_dir_ml_labeled_all = f'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\ML\labled_all'
dataframe_dir_ml_labeled_m = f'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\ML\labled_first_more'


# df_sessions = pd.read_pickle(fr'{dataframe_dir_users}\user-sessions_features_labeled.pickle')

def svm_classifier(x, y, filename, report_df):
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

    print("-----report----------")
    # print(metrics.confusion_matrix(y_test,y_predict))
    print(metrics.classification_report(y_test, y_predict))  # output_dict=True))
    report = pd.DataFrame.from_dict(metrics.classification_report(y_test, y_predict, output_dict=True))
    report['target'] = filename
    report['algorithm'] = "SVM"

    print("-----score----------")
    score = metrics.accuracy_score(y_test, y_predict)
    print(score)
    report['score'] = score

    print("---------confusion matrix--------")
    cm = confusion_matrix(y_test, y_predict)
    print(cm)

    return pd.concat([report_df, report])
    # print("-----importance----------")
    #
    # perm_importance = permutation_importance(svclassifier, x_test, y_test, scoring='accuracy')
    # importance = perm_importance.importances_mean
    # for i,v in enumerate(importance):
    #     print('Feature: %0d, Score: %.5f' % (i,v))
    #
    # features = np.array(feature_names)
    #
    # sorted_idx = importance.argsort()
    # plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
    # plt.xlabel("Permutation Importance")


def naive_bayes_classifier(x, y, filename, report_df):
    print("naive bayes")
    labels = y.unique()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    NB = GaussianNB()
    NB.fit(x_train, y_train)

    # Predict Output
    y_predict = NB.predict(x_test)

    print("-----report----------")
    print(metrics.classification_report(y_test, y_predict))  # output_dict=True))
    report = pd.DataFrame.from_dict(metrics.classification_report(y_test, y_predict, output_dict=True))
    report['target'] = filename
    report['algorithm'] = "naive_bayes"

    print("-----score----------")
    score = metrics.accuracy_score(y_test, y_predict)
    print(score)
    report['score'] = score

    print("---------confusion matrix--------")
    cm = confusion_matrix(y_test, y_predict)
    print(cm)

    return pd.concat([report_df, report])


def kNeighbors_classifier(x, y, filename, report_df):
    print('***KNeighborsClassifier***')

    labels = y.unique()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    KNN = KNeighborsClassifier(n_neighbors=3)
    # fit the model
    KNN.fit(x_train, y_train)

    # Predict Output
    y_predict = KNN.predict(x_test)

    print("-----report----------")
    print(metrics.classification_report(y_test, y_predict))  # output_dict=True))
    report = pd.DataFrame.from_dict(metrics.classification_report(y_test, y_predict, output_dict=True))
    report['target'] = filename
    report['algorithm'] = "KNN"

    print("-----score----------")
    score = metrics.accuracy_score(y_test, y_predict)
    print(score)
    report['score'] = score

    print("---------confusion matrix--------")
    cm = confusion_matrix(y_test, y_predict)
    print(cm)

    return pd.concat([report_df, report])
    # print("-----importance----------")
    #
    # # perform permutation importance
    # # results = permutation_importance(KNN, x_train, y_train, scoring='accuracy')
    # results = permutation_importance(KNN, x_test, y_test, scoring='accuracy')
    # # get importance
    # importance = results.importances_mean
    # # summarize feature importance
    # for i,v in enumerate(importance):
    #     print('Feature: %0d, Score: %.5f' % (i,v))
    #
    # # plot feature importance
    # plt.bar([x for x in range(len(importance))], importance)
    # plt.show()


def random_forest_classifier(x, y, filename, report_df):
    print('***Random Forest***')

    feature_list = x.columns  # list(x_features.columns)
    x = np.array(x)
    labels = df_sessions['target_label'].unique()
    y = np.array(y)

    x_train_features, x_test_features, y_train_labels, y_test_labels = train_test_split(x, y, test_size=0.25, random_state=42)
    # print('Training Features Shape:', x_train_features.shape)
    # print('Training Labels Shape:', y_train_labels.shape)
    # print('Testing Features Shape:', x_test_features.shape)
    # print('Testing Labels Shape:', y_test_labels.shape)

    forest = RandomForestClassifier(criterion='gini', n_estimators=5, random_state=42, n_jobs=2, class_weight='balanced') # TODO class_weight='balanced_subsample' or balanced
    # BalancedRandomForestClassifier(n_estimators=10)
    forest.fit(x_train_features, y_train_labels)

    y_predict = forest.predict(x_test_features)

    print("-----report----------")
    print(metrics.classification_report(y_test_labels, y_predict))  # output_dict=True))
    report = pd.DataFrame.from_dict(metrics.classification_report(y_test_labels, y_predict, output_dict=True))
    report['target'] = filename
    report['algorithm'] = "random_forest"

    print("-----score----------")
    score = metrics.accuracy_score(y_test_labels, y_predict)
    print(score)
    report['score'] = score

    print("---------confusion matrix--------")
    cm = confusion_matrix(y_test_labels, y_predict)
    print(cm)

    print("-------importance--------")
    importance = pd.DataFrame({'feature': feature_list, 'importance': np.round(forest.feature_importances_, 3)})
    importance.sort_values('importance', ascending=False, inplace=True)
    print(importance)
    importance.to_csv(fr'{dataframe_dir_ml}\feature_importance\{filename}-randomForest_f_importance.csv')
    # with open(fr'{dataframe_dir_ml}\feature_importance\{filename}-decision_tree_f_importance.txt', 'w') as f:
    #     #print(importance, file=f)
    #     f.write(importance)

    return pd.concat([report_df, report])

    # print('--------Precision-------')
    # precision = metrics.precision_score(y_test_labels, y_predict, average=None)
    # precision_results = pd.DataFrame(precision, index=labels)
    # precision_results.rename(columns={0: 'precision'}, inplace=True)
    # print(precision_results)


def decision_tree_classifier(x, y, filename, report_df):
    print('***Decision tree***')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    clf_model = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42, min_samples_leaf=5)
    clf_model.fit(x_train, y_train)

    # tree_rules = export_text(clf_model, feature_names=list(feature_names))
    # print(tree_rules)

    y_predict = clf_model.predict(x_test)

    print("-----report----------")
    print(metrics.classification_report(y_test, y_predict))  # output_dict=True), file=f)
    report = pd.DataFrame.from_dict(metrics.classification_report(y_test, y_predict, output_dict=True))
    report['target'] = filename
    report['algorithm'] = "decision_tree"  # 'str' + df['col'].astype(str)

    print("-----score----------")
    score = metrics.accuracy_score(y_test, y_predict)
    print(score)
    report['score'] = score

    print("---------confusion matrix--------")
    cm = confusion_matrix(y_test, y_predict)
    print(cm)

    print("-------importance--------")
    importance = pd.DataFrame({'feature': x_train.columns, 'importance': np.round(clf_model.feature_importances_, 3)})
    importance.sort_values('importance', ascending=False, inplace=True)
    importance.reset_index(drop=True, inplace=True)
    print(importance)
    importance.to_csv(fr'{dataframe_dir_ml}\feature_importance\{filename}-decision_tree_f_importance.csv')
    # with open(fr'{dataframe_dir_ml}\feature_importance\{filename}-decision_tree_f_importance.txt', 'w') as f:
    #     print(importance, file=f)
    print(type(report_df))
    print(type(report))
    return pd.concat([report_df, report])


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    # df_sessions = pd.read_pickle(fr'{dataframe_dir_ml}\user-sessions_features_labeled.pickle')
    # x, y = ML_helpers.prepare_df_no_oversampling(df_sessions)

    pathlist = pathlib.Path(dataframe_dir_ml_labeled).glob('**/*.pickle')
    report_all = pd.DataFrame()

    for data_path in pathlist:
        path_in_str = str(data_path)

        # with open(f'{dataframe_dir_ml_labeled}\outout_noOversampling.txt', 'w') as f:
        print(f'###################  target: {data_path.stem}   #############################')  # , file=f)

        df_sessions = pd.read_pickle(path_in_str)
        df_sessions.to_csv(fr'{dataframe_dir_ml_labeled}\test_{data_path.stem}.csv')
        x, y = ML_helpers.prepare_df_undersampling(df_sessions)

        report_all = decision_tree_classifier(x, y, data_path.stem, report_all)

        # report_all = random_forest_classifier(x, y, data_path.stem, report_all)

        # report_all = kNeighbors_classifier(x, y, data_path.stem, report_all)

        # report_all = naive_bayes_classifier(x, y, data_path.stem, report_all)

        # x, y = ML_helpers.prepare_df_no_oversampling(df_sessions)
        # report_all = svm_classifier(x, y, data_path.stem, report_all)
        # h = input()
        # if h == 'x':
        #     break
        # else:
        #     continue

    # report_all.to_csv(fr'{dataframe_dir_ml}\report_ml_undersampling_combined.csv')
