from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

dataframe_dir_ml = r'./../RabbitHoleProcess/data/dataframes/ML'
milkGreen = '#0BCB85'

path_categories = '../../RabbitHoleProcess/data/categories/app_categorisation_2020.csv'
df_categories = pd.read_csv(path_categories, sep=';')
df_categories.drop(columns=['Perc_users', 'Training_Coding_1', 'Training_Coding_all', 'Training_Coding_2', 'Rater1', 'Rater2'], inplace=True)


def get_app_category(app_packagename):
    """
    Get tha app category for an application package with the help of a app category dataset
    :param app_packagename: the package name
    :return: the app category
    """
    category = df_categories[(df_categories['App_name'].values == app_packagename)]['Final_Rating']
    if not category.empty:
        return category.values[0]
    else:
        return 'UNKNOWN'


def check_labels():
    """
    Get the size of the two classification classes
    """
    df_sessions = pd.read_pickle(fr'{dataframe_dir_ml}\user-sessions_features_labeled.pickle')
    df_sessions = clean_df(df_sessions)
    # df_sessions.to_csv(fr'{dataframe_dir_users}\user-sessions_features_labeled_cleand.csv')
    sns.countplot(df_sessions['target_label'], color=milkGreen)
    print(len(df_sessions[df_sessions['target_label'] == 'rabbit_hole']))
    print(len(df_sessions[df_sessions['target_label'] == 'no_rabbithole']))
    plt.show()


def undersampling(df_x_features, df_y_labels):
    print('-----undersampling----')
    rus = RandomUnderSampler(random_state=42, replacement=True)  # fit predictor and target variable
    x_rus, y_rus = rus.fit_resample(df_x_features, df_y_labels)

    print('original dataset shape:', Counter(df_y_labels))
    print('Resample dataset shape', Counter(y_rus))
    return x_rus, y_rus


def oversampling(df_x_features, df_y_labels):
    print('------oversampling--------')

    ros = RandomOverSampler(random_state=42)
    # fit predictor and target variable
    x_ros, y_ros = ros.fit_resample(df_x_features, df_y_labels)

    print('Original dataset shape', Counter(df_y_labels))
    print('Resample dataset shape', Counter(y_ros))
    return x_ros, y_ros


def oversampling_smote(df_x_features, df_y_labels):
    print('----oversampling smote-----')
    smote = SMOTE(random_state=42)

    # fit predictor and target variable
    x_smote, y_smote = smote.fit_resample(df_x_features, df_y_labels)

    print('Original dataset shape', Counter(df_y_labels))
    print('Resample dataset shape', Counter(y_smote))
    return x_smote, y_smote


def clean_df(df):
    """
    Prepare the df for machine learning and drop all unnecessary columns
    :param df: the dataframe to clean
    :return: the cleand df
    """
    # return df.drop(columns=['session_id', 'studyID', 'session_length', 'timestamp_1', 'timestamp_2', 'count', 'f_sequences_apps', 'f_esm_intention', 'f_bag_of_apps']).fillna(0)
    df.drop(df.index[df['f_session_group_timespan'].isnull()], inplace=True)
    return df.drop(columns=['group_id','session_ids', 'studyID', 'timestamp_1', 'timestamp_2', 'count', 'f_sequences_apps', 'f_bag_of_apps']).fillna(0)


def prepare_df_oversampling(df):
    print("--prepare df---")
    df = clean_df(df)
    x = df.drop('target_label', axis=1)
    y = df['target_label']
    # save the feature name and target variables

    return oversampling_smote(x, y)


def prepare_df_undersampling(df):
    print("--prepare df---")
    df = clean_df(df)
    x = df.drop('target_label', axis=1)
    y = df['target_label']
    # save the feature name and target variables
    return undersampling(x, y)


def prepare_df_no_oversampling(df):
    print("--prepare df not oversampling---")
    df = clean_df(df)

    x = df.drop('target_label', axis=1)
    y = df['target_label']

    return x, y



