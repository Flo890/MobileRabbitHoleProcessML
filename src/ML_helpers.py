from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

dataframe_dir_users = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\users'


def check_labels():
    df_sessions = pd.read_pickle(fr'{dataframe_dir_users}\user-sessions_features_labeled.pickle')
    df_sessions = clean_df(df_sessions)
    # df_sessions.to_csv(fr'{dataframe_dir_users}\user-sessions_features_labeled_cleand.csv')
    sns.countplot(df_sessions['target_label'])
    print(len(df_sessions[df_sessions['target_label'] == 'rabbit_hole']))  # 430
    print(len(df_sessions[df_sessions['target_label'] == 'no_rabbithole']))  # 15408
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
    smote = SMOTE()

    # fit predictor and target variable
    x_smote, y_smote = smote.fit_resample(df_x_features, df_y_labels)

    print('Original dataset shape', Counter(df_y_labels))
    print('Resample dataset shape', Counter(y_smote))
    return x_smote, y_smote


def clean_df(df):
    return df.drop(columns=['session_id', 'studyID', 'session_length', 'timestamp_1', 'timestamp_2', 'count', 'f_sequences_apps', 'f_esm_intention', 'f_bag_of_apps']).fillna(0)


def prepare_clustering(df):
    return df[['f_session_length', 'f_bag_of_apps']].fillna(0)

def prepare_df(df):
    print("--prepare df---")
    df = clean_df(df)
    x = df.drop('target_label', axis=1)
    y = df['target_label']
    # save the feature name and target variables

    return oversampling_smote(x, y)


def prepare_df_no_oversampling(df):
    print("--prepare df not oversampling---")
    df = clean_df(df)

    x = df.drop('target_label', axis=1)
    y = df['target_label']

    return x, y


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    check_labels()
