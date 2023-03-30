import math

import pandas as pd
import numpy as np
import pathlib
import pickle
import getFromJson
import prepareTimestamps
import extractSessions
import featureExtraction
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels import robust
from sklearn.preprocessing import OneHotEncoder

raw_data_user = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\live\2022-04-27T20 20 31Z_absentmindedtrack-default-rtdb_data.json.gz'
raw_data_live = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\live'
logs_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\logFiles'
user_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\cleanedUsers'
dataframe_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes'

dataframe_dir_live = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\live'

dataframe_dir_users = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\users'
dataframe_dir_ml = r'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessions_ml'
dataframe_dir_sessions = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\sessions'
dataframe_dir_sessions_features = r'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessions_with_features'
dataframe_dir_bagofapps_vocab = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\bag_of_apps_vocabs'

dataframe_dir_live_logs_sorted = r'D:\usersorted_logs_preprocessed'
questionnaires_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData'

dataframe_dir_ml_labeled = f'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\ML\labled_data'
clean_users_dir_path = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\users'


def extractData():
    """
    Extract the logs for each user from the raw data files
    """
    # only run once
    # getFromJson.extractUsers(file_path=raw_data_live, end_directory=dataframe_dir_users, is_gzip=True)

    # extract logs for files
    getFromJson.extract_logs(directory=raw_data_live, end_directory=dataframe_dir_live, save_type=3, is_gzip=True)

    # concat all extracted logs
    # getFromJson.concat_user_logs(logs_dic_directory=dataframe_dir_live, end_directory=dataframe_dir_live_logs_sorted)

    user_list = getFromJson.getStudIDlist(clean_users_dir_path)

    for item in user_list:
        print(item)
        getFromJson.concat_logs_one_user(logs_dic_directory=dataframe_dir_live, end_directory=dataframe_dir_live_logs_sorted, studyID=item)


def concat_additional_logs():
    """
    Add extracted log files later participants form database files
    :return:
    """
    print("çoncat logs")
    pathlist = pathlib.Path(dataframe_dir_live_logs_sorted).glob('**/*.pickle')

    for data_path in pathlist:
        path_in_str = str(data_path)
        print(f"read dic: {path_in_str}")
        df_logs = pd.read_pickle(path_in_str)

        df_logs.sort_values(by=['correct_timestamp'], ascending=True, inplace=True)

        # for data_path_old in pathlist_old:
        #     print(str(data_path_old))
        #     print(data_path.stem, data_path_old.stem)
        #     if data_path.stem == data_path_old.stem:
        #         print('same')
        #         df_logs_old = pd.read_pickle(str(data_path_old))
        #         print(df_logs.columns.values.size)
        #         print(df_logs_old.columns.values.size)
        #         df_concat = pd.concat([df_logs_old, df_logs], ignore_index=False)
        #         with open(fr'{dataframe_dir_live_logs_sorted}\{data_path.stem}.pickle', 'wb') as f:
        #             pickle.dump(df_concat, f, pickle.HIGHEST_PROTOCOL)
        #
        # df_logs = pd.read_pickle(path_in_str)
        # df_logs = cleanup(df_logs)
        # df_logs = prepareTimestamps.process_timestamps(df_logs)
        # # extract Metadata
        # df_logs = getFromJson.extractMetaData(df_logs)

        with open(fr'{dataframe_dir_live_logs_sorted}\{data_path.stem}.pickle', 'wb') as f:
            pickle.dump(df_logs, f, pickle.HIGHEST_PROTOCOL)


def extractUser():
    getFromJson.extractUsers(file_path=raw_data_user, end_directory=clean_users_dir_path, is_gzip=True)


def cleanup(df_logs):
    """
    remove unnessesary logs
    :param df_logs: the dataframe to clean
    :return: the cleaned dataframe
    """
    print('_clean up_')

    # get rid of RabbitHole events aka admin and packagename com.lmu.rabbitholetracker
    df_logs.drop(df_logs.index[df_logs['packageName'] == 'com.lmu.trackingapp'], inplace=True)

    # Remove overhead logs aka TYPE_WINDOW_CONTENT_CHANGED
    df_logs.drop(df_logs.index[df_logs['event'] == 'TYPE_WINDOW_CONTENT_CHANGED'], inplace=True)

    return df_logs


def preprocessing():
    """
    Run all necessary preprocessing steps,
     prepare timestamps, flattens metadata, extract user sessions, device info and installed apps
    """
    path_list = pathlib.Path(dataframe_dir_live_logs_sorted).glob('**/*.pickle')

    dic_device = []
    dic_installed_apps = {}

    # user_sessions = {}
    # list_user_sessions_all = []

    for data_path in path_list:
        path_in_str = str(data_path)
        df_logs = pd.read_pickle(path_in_str)
        print(f'preprocess {data_path.stem}')

        df_logs = cleanup(df_logs)
        df_logs = prepareTimestamps.process_timestamps(df_logs)
        # extract Metadata
        df_logs = getFromJson.extractMetaData(df_logs)

        # extract installed apps and dive info -> might add to user df
        # DEVICE_INFO
        device = df_logs[df_logs['event'] == 'DEVICE_INFO']
        if not device.empty:
            info = device[0]
            dic_device.append({'studyID': data_path.stem, 'model': info.event, 'releaseVersion': info.description, 'sdkVersion': info.name, 'manufacturer': info.packageName})

        # LogEventName.DEVICE_INFO, timestamp, event= model, description = releaseVersion, name = sdkVersion.toString(), packageName = manufacturer )
        # INSTALLED_APP
        apps = df_logs[df_logs['event'] == 'INSTALLED_APP']
        if not apps.empty:
            dic_installed_apps[data_path.stem] = df_logs[df_logs['event'] == 'INSTALLED_APP']['packageName']

        extracted = extractSessions.extract_sessions(df_logs)
        # user_sessions[data_path.stem] = extracted[1]
        # list_user_sessions_all.append(extracted[1])

        with open(fr'{dataframe_dir_live_logs_sorted}\{data_path.stem}.pickle', 'wb') as f:
            pickle.dump(extracted[0], f, pickle.HIGHEST_PROTOCOL)
            f.close()

        with open(fr'{dataframe_dir_sessions}\{data_path.stem}-sessions.pickle', 'wb') as f:
            pickle.dump(extracted[1], f, pickle.HIGHEST_PROTOCOL)

    # user_sessions_all = pd.concat(list_user_sessions_all, ignore_index=False)
    # user_sessions_all.to_csv(fr'{dataframe_dir_live}\user_sessions_all.csv')
    # with open(fr'{dataframe_dir_live}\user_sessions_all.pickle', 'wb') as f:
    #     pickle.dump(user_sessions, f, pickle.HIGHEST_PROTOCOL)

    with open(fr'{clean_users_dir_path}\user-device_info.pickle', 'wb') as f:
        pickle.dump(pd.DataFrame(dic_device), f, pickle.HIGHEST_PROTOCOL)

    with open(fr'{clean_users_dir_path}\user-installed_apps.pickle', 'wb') as f:
        pickle.dump(dic_installed_apps, f, pickle.HIGHEST_PROTOCOL)


def extract_features():
    """
    Extract the features for each user from the log file and add it to the sessions and feature dataframe
    :return: saves the session list to file
    """
    path_list = pathlib.Path(dataframe_dir_live_logs_sorted).glob('**/*.pickle')
    # df_all_sessions = pd.read_pickle(path_testfile_sessions)
    # print(df_all_sessions)

    for data_path in path_list:
        path_in_str = str(data_path)
        # df_all_logs = pd.read_pickle(path_in_str)
        print('session_file', f'{dataframe_dir_sessions}\{data_path.stem}-sessions.pickle')
        df_all_sessions = pd.read_pickle(f'{dataframe_dir_sessions}\{data_path.stem}-sessions.pickle')

        print(data_path.stem)
        # session_for_id = df_all_sessions[data_path.stem]

        if not df_all_sessions.empty:
            print("not empty")
            df_session_features = featureExtraction.get_features_for_session(df_logs=pd.read_pickle(path_in_str), df_sessions=df_all_sessions)
            df_all_sessions = df_session_features[0]

            with open(fr'{dataframe_dir_bagofapps_vocab}\{data_path.stem}-vocab.pickle', 'wb') as f:
                pickle.dump(df_session_features[1], f, pickle.HIGHEST_PROTOCOL)

        with open(fr'{dataframe_dir_sessions_features}\{data_path.stem}.pickle', 'wb') as f:
            pickle.dump(df_all_sessions, f, pickle.HIGHEST_PROTOCOL)


def check_for_activity():
    path_list = pathlib.Path(dataframe_dir_live_logs_sorted).glob('**/*.pickle')
    for data_path in path_list:
        path_in_str = str(data_path)
        # df_all_logs = pd.read_pickle(path_in_str)

        print(data_path.stem)
        # session_for_id = df_all_sessions[data_path.stem]
        result = featureExtraction.checkforactivity(pd.read_pickle(path_in_str))
        print(result)


def filter_sessions_esm_user_based():
    """
    Filter the sessions with ESM and above 45s, calculate and save stats, remove outliners
    :return:
    """
    # Filter the relevant sessions
    # Get all sessions that are over 45s and have a esm answers (f_esm_finished_intention not empty)
    threshold = pd.Timedelta(seconds=45)
    threshold_max_cap = pd.Timedelta(seconds=25200)  # 7 stunden

    session_stats = []
    sessionlength = []

    path_list = pathlib.Path(dataframe_dir_sessions_features).glob('**/*.pickle')
    for data_path in path_list:
        path_in_str = str(data_path)
        print(data_path.stem)

        df_session_features = pd.read_pickle(path_in_str)
        if not df_session_features.empty:
            # print(df_session_features['f_session_length'].astype('timedelta64[s]').head(), len(df_session_features))
            # testtt = df_session_features['f_session_length'].values.astype('timedelta64[s]')

            # ax = sns.stripplot(x= (df_session_features['f_session_length'] / pd.Timedelta(seconds=1)))
            sessionlength.append(df_session_features['f_session_length'])
            print(len(df_session_features['f_session_length']))
            print('typeeee', type(df_session_features['f_session_length'].values[0]))
            print(df_session_features['f_session_length'].head())

            upper_limit = df_session_features['f_session_length'].quantile(0.99)
            lower_limit = df_session_features['f_session_length'].quantile(0.01)

            print(upper_limit)
            print(lower_limit)

            # drop sessions
            # df_session_features = df_session_features.drop(df_session_features['f_session_length'] > threshold_max_cap)
            df_filtered_esm = df_session_features[(df_session_features['f_session_length'] > threshold) & (df_session_features['f_esm_finished_intention'] != '')]

            with open(fr'{dataframe_dir_sessions}\{data_path.stem}-sessions_filtered.pickle', 'wb') as f:
                pickle.dump(df_filtered_esm, f, pickle.HIGHEST_PROTOCOL)

            df_esm_rabbithole_finished = df_filtered_esm[(df_filtered_esm['f_esm_finished_intention'] == 'No')]
            df_esm_rabbithole_more = df_filtered_esm[(df_filtered_esm['f_esm_more_than_intention'] == 'Yes')]
            df_esm_rabbithole_finished_more = df_filtered_esm[(df_filtered_esm['f_esm_finished_intention'] == 'No') & (df_filtered_esm['f_esm_more_than_intention'] != 'Yes')]

            session_length_mean = pd.to_timedelta(df_session_features['f_session_length']).mean()

            session_stats.append({'studyID': data_path.stem, 'session_count': len(df_session_features), 'esm_over45s_count': len(df_filtered_esm), 'session_length_mean': session_length_mean,
                                  'sessions_not_finished': len(df_esm_rabbithole_finished),
                                  'session_more_than_intention': len(df_esm_rabbithole_more), 'sessions_not_finished_and_more': len(df_esm_rabbithole_finished_more)})

    sessions_list = pd.DataFrame(session_stats)
    sessions_list.to_csv(fr'{user_dir}\sessions_stats.csv')


def filter_sessions_outliners_all():
    """
    Filter and remove the outliner sessions
    :return:
    """

    feature_path = fr'{dataframe_dir_ml}\user-sessions_features_all.pickle'
    df_all_sessions = pd.read_pickle(feature_path)

    # sns.distplot(df_session_features['f_session_length'] / pd.Timedelta(milliseconds=1))
    # plt.show()
#    sns.histplot(df_all_sessions['f_session_length'])
  #  plt.show()

    df_all_sessions_s = df_all_sessions['f_session_length']
    # remove sessions without app
    df_all_sessions_filtered1 = df_all_sessions#[df_all_sessions['f_sequences_apps'].notnull()]

    # zscore https://medium.com/clarusway/z-score-and-how-its-used-to-determine-an-outlier-642110f3b482
    # https://stats.stackexchange.com/questions/16822/how-to-log-transform-z-scores
    df_all_sessions_filtered1['f_session_length_log'] = np.log(df_all_sessions_filtered1['f_session_length']+1)
    df_all_sessions_filtered1['zscore'] = (df_all_sessions_filtered1['f_session_length_log'] - df_all_sessions_filtered1['f_session_length_log'].mean()) / df_all_sessions_filtered1['f_session_length_log'].std(ddof=0)
    # https://towardsdatascience.com/outlier-detection-part1-821d714524c
    # https://www.statology.org/median-absolute-deviation-in-python/
    df_all_sessions_filtered1['zscore_mod'] = (df_all_sessions_filtered1['f_session_length_log'] -
                                           df_all_sessions_filtered1['f_session_length_log'].mean()) / \
                                          robust.mad(df_all_sessions_filtered1['f_session_length_log'])

    df_all_sessions_filtered1["is_outlier"] = (abs(df_all_sessions_filtered1["zscore"])>3).astype(int) # = sessions longer than 50 minutes
    df_sessions_filtered = df_all_sessions_filtered1[(df_all_sessions_filtered1['is_outlier'] == 0)]

    # quantile LIMIT --------------------------------------
  #  upper_limit = df_all_sessions['f_session_length'].quantile(0.996)  # bis 5.5h bei 0.995 bis 3.5h
   # lower_limit = df_all_sessions['f_session_length'].quantile(0.01)

   # df_sessions_filtered = df_all_sessions[(df_all_sessions['f_session_length'] <= upper_limit) & (df_all_sessions['f_session_length'] >= lower_limit)]
 #   print(type(df_sessions_filtered))
    plt.hist(df_sessions_filtered['f_session_length_log'])
    plt.show()

#    print(upper_limit)
#    print(lower_limit)

    with open(fr'{dataframe_dir_ml}\user-sessions_features_all_of.pickle', 'wb') as f:
        pickle.dump(df_sessions_filtered, f, pickle.HIGHEST_PROTOCOL)


def create_labels_single():
    """
    Label the session with the target class, using a single ES features as target
    """
    print('create label')

    df_sessions = pd.read_pickle(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle')
    df_sessions.reset_index(drop=True, inplace=True)
    rh = 'rabbit_hole'
    no_rh = 'no_rabbithole'

    df_sessions.insert(6, 'target_label', '')

    for index, session in enumerate(df_sessions.itertuples(index=False)):

        # ['f_esm_intention', 'f_esm_finished_intention', 'f_esm_more_than_intention', 'f_esm_track_of_time', 'f_esm_track_of_space', 'f_esm_emotion', 'f_esm_regret', 'f_esm_agency']
        if df_sessions.loc[index, 'f_esm_more_than_intention_Yes'] == 1:
            df_sessions.at[index, 'target_label'] = rh
        else:
            df_sessions.at[index, 'target_label'] = no_rh

    # df_sessions = drop_esm_features(df_sessions)

    with open(fr'{dataframe_dir_ml_labeled}\sessions_features_labeled_more_than_intention_with_esm.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def labeling_combined():
    """
    Label the session with a target class, using a combination of features as target
    :return:
    """
    labelist = [('f_esm_more_than_intention_Yes', 'f_esm_emotion_interest-expectancy'), ('f_esm_more_than_intention_Yes', 'f_esm_emotion_happiness-elation'),
                ('f_esm_more_than_intention_Yes', 'f_esm_emotion_calm-contentment')] \
        # ('f_esm_more_than_intention_Yes', 'f_esm_emotion_boredom-indifference'), ('f_esm_more_than_intention_Yes', 'f_esm_emotion_anxiety-fear')]
    labelListScale = [('f_esm_more_than_intention_Yes', 'f_esm_regret_7.0', 'f_esm_regret_6.0', 'f_esm_regret_5.0'),
                      ('f_esm_more_than_intention_Yes', 'f_esm_agency_0.0', 'f_esm_agency_1.0', 'f_esm_agency_2.0'),
                      ('f_esm_more_than_intention_Yes', 'f_esm_track_of_time_6.0', 'f_esm_track_of_time_7.0', 'f_esm_track_of_time_5.0'),
                      ('f_esm_more_than_intention_Yes', 'f_esm_track_of_space_0.0', 'f_esm_track_of_space_1.0', 'f_esm_track_of_space_2.0')]

    for item in labelist:
        create_labels_combines(item[0], item[1])

    for item in labelListScale:
        create_labels_combines_scale(item[0], item[1], item[2], item[3])


def create_labels_combines(label1, label2):
    """
    Helper function for combined target labels
    """
    print('create label')

    df_sessions = pd.read_pickle(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle')
    df_sessions.reset_index(drop=True, inplace=True)
    # df_sessions.to_csv(fr'{dataframe_dir_ml}\user-sessions_features_all_t.pickle')
    rh = 'rabbit_hole'
    no_rh = 'no_rabbithole'

    # df_sessions['f_bag_of_apps'] = pd.Series(dtype='string')
    df_sessions.insert(6, 'target_label', '')

    for index, session in enumerate(df_sessions.itertuples(index=False)):
        if (df_sessions.loc[index, label1] == 1) & (df_sessions.loc[index, label2] == 1):
            df_sessions.at[index, 'target_label'] = rh
        else:
            df_sessions.at[index, 'target_label'] = no_rh

    df_sessions = drop_esm_features(df_sessions)

    with open(fr'{dataframe_dir_ml}\labled\user-sessions_features_labeled_{label1}_{label2}.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def create_labels_combines_scale(label1, label2_1, label2_2, label2_3):
    """
    Helper function for combined target labels
    """
    print('create label')

    df_sessions = pd.read_pickle(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle')
    df_sessions.reset_index(drop=True, inplace=True)
    rh = 'rabbit_hole'
    no_rh = 'no_rabbithole'

    df_sessions.insert(6, 'target_label', '')

    for index, session in enumerate(df_sessions.itertuples(index=False)):
        if (df_sessions.loc[index, label1] == 1) & ((df_sessions.loc[index, label2_1] == 1) | (df_sessions.loc[index, label2_2] == 1) | (df_sessions.loc[index, label2_3] == 1)):
            df_sessions.at[index, 'target_label'] = rh
        else:
            df_sessions.at[index, 'target_label'] = no_rh

    # df_sessions = drop_esm_features(df_sessions)

    with open(fr'{dataframe_dir_ml}\labled\user-sessions_features_labeled_{label1}_{label2_1}.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def drop_esm_features(df_sessions):
    """
    Drop all ESM feature colums form the dataframe
    :param df_sessions: the dataframe to drop
    :return:
    """
    colums_esm = [x for x in df_sessions.columns.values if x.startswith('f_esm')]
    df_sessions.drop(columns=colums_esm, inplace=True)
    return df_sessions


def one_hot_encoding_dummies():
    """
    One-hot-encode features using pandas.dummies
    :return:
    """
    print('on hot encoding dummies')
    df_sessions = pd.read_pickle(fr'{dataframe_dir_users}\user-sessions_features_all.pickle')
    # df_sessions.to_csv(fr'{dataframe_dir_users}\user-sessions_features_all.csv')
    # replace empty cells
    df_sessions = df_sessions.replace(r'^\s*$', np.nan, regex=True)
    to_encode = ['f_demographics_gender', 'f_esm_finished_intention', 'f_esm_more_than_intention', 'f_esm_emotion', 'f_esm_track_of_time', 'f_esm_track_of_space', 'f_esm_regret', 'f_esm_agency',
                 'f_hour_of_day', 'f_weekday']

    for column in to_encode:
        df_encoded = pd.get_dummies(df_sessions[column], prefix=column)
        df_sessions.drop(columns=[column], inplace=True)
        df_sessions = pd.concat([df_sessions, df_encoded], axis=1)

    with open(fr'{dataframe_dir_users}\user-sessions_features_1hotencoded', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def one_hot_encoding_scilearn():
    """
    One-hot-encode features using scilearn OneHotEncoder
    :return:
    """
    print('one hot encoding scilearn')
    df_sessions = pd.read_pickle(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle')
    df_sessions = df_sessions.replace(r'^\s*$', np.nan, regex=True)
    enc = OneHotEncoder(handle_unknown='ignore')

    to_encode = ['f_demographics_gender',
         'f_esm_finished_intention', 'f_esm_more_than_intention', 'f_esm_emotion', 'f_esm_track_of_time', 'f_esm_track_of_space', 'f_esm_regret', 'f_esm_agency',
                 'f_hour_of_day', 'f_weekday']

    for column in to_encode:
        # passing bridge-types-cat column (label encoded values of bridge_types)
        end = enc.fit_transform(df_sessions[[column]]).toarray()
        column_name = enc.get_feature_names_out([column])
        enc_df = pd.DataFrame(end, columns=column_name)

        # merge with main df bridge_df on key values
        # df_sessions = df_sessions.join(enc_df)
        df_sessions = pd.concat([df_sessions, enc_df], axis=1).drop(columns=[column])

    # df_sessions.to_csv(fr'{dataframe_dir_users}\user-sessions_features_encoded_sci.csv')

    with open(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def demographics_encode_age():
    """
    Encode the age features into bag of ages
    :return:
    """
    print("Encode age")
    # df_sessions = pd.read_pickle(fr'{dataframe_dir_ml}\labled_data\user-sessions_features_labeled_more_than_intention.pickle')
    df_sessions = pd.read_pickle(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle')

    # Bag Age in age groups
    df_sessions['f_demographics_age_0_19'] = 0
    df_sessions['f_demographics_age_20_29'] = 0
    df_sessions['f_demographics_age_30_39'] = 0
    df_sessions['f_demographics_age_40_49'] = 0
    df_sessions['f_demographics_age_50_59'] = 0
    df_sessions['f_demographics_age_60_69'] = 0
    df_sessions['f_demographics_age_70_100'] = 0

    for index, session in enumerate(df_sessions.itertuples(index=False)):
        if df_sessions.loc[index, 'f_demographics_age'] <= 19:
            df_sessions.at[index, 'f_demographics_age_0_19'] = 1
        elif (df_sessions.loc[index, 'f_demographics_age'] <= 29) & (df_sessions.loc[index, 'f_demographics_age'] >= 20):
            df_sessions.at[index, 'f_demographics_age_20_29'] = 1
        elif (df_sessions.loc[index, 'f_demographics_age'] <= 39) & (df_sessions.loc[index, 'f_demographics_age'] >= 30):
            df_sessions.at[index, 'f_demographics_age_30_39'] = 1
        elif (df_sessions.loc[index, 'f_demographics_age'] <= 49) & (df_sessions.loc[index, 'f_demographics_age'] >= 40):
            df_sessions.at[index, 'f_demographics_age_40_49'] = 1
        elif (df_sessions.loc[index, 'f_demographics_age'] <= 59) & (df_sessions.loc[index, 'f_demographics_age'] >= 50):
            df_sessions.at[index, 'f_demographics_age_50_59'] = 1
        elif (df_sessions.loc[index, 'f_demographics_age'] <= 69) & (df_sessions.loc[index, 'f_demographics_age'] >= 60):
            df_sessions.at[index, 'f_demographics_age_60_69'] = 1
        elif df_sessions.loc[index, 'f_demographics_age'] >= 70:
            df_sessions.at[index, 'f_demographics_age_70_100'] = 1

    df_sessions.drop(columns='f_demographics_age', inplace=True)

    # with open(fr'{dataframe_dir_ml}\labled_data\user-sessions_features_all_age_bags.pickle', 'wb') as f:
    with open(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def bag_of_apps_create_vocab():
    """
    Create the vocabulary for the bag of apps method
    :return:
    """
    print('create bag of apps vocabulary')
    # get the vocabulary, which is every application package used by every user?
    # create vocab from every users app list
    path_list = pathlib.Path(dataframe_dir_bagofapps_vocab).glob('**/*.pickle')
    vocab = []
    for data_path in path_list:
        path_in_str = str(data_path)
        print(path_in_str)
        user_vocab = pd.read_pickle(path_in_str)
        vocab = list(set(vocab + user_vocab))

    print(len(vocab))
    print(vocab)
    with open(fr'{dataframe_dir_ml}\bag_of_apps_vocab.pickle', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


def bag_of_apps_create_bags():
    """
    Create a bag of apps vector for each session in the df of all sessions
    :return:
    """
    print('create bag of apps')
    # go over every session and create vector of app count
    vocab = pd.read_pickle(fr'{dataframe_dir_ml}\bag_of_apps_vocab.pickle')
    df_sessions = pd.read_pickle(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle')

    df_sessions['f_bag_of_apps'] = pd.Series(dtype='object')
    df_sessions['f_sequences_apps'].fillna(" ", inplace=True)

    for index, session in enumerate(df_sessions.itertuples(index=False)):
        bag_vector = [0] * len(vocab)

        app_list = session.f_sequences_apps[0]
        for a in app_list:
            for i, app in enumerate(vocab):
                if app == a:
                    bag_vector[i] += 1

        array = np.array([object])
        array[0] = bag_vector
        df_sessions.at[index, 'f_bag_of_apps'] = bag_vector

    with open(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def convert_timedeletas():
    """
    Convert time deltas to seconds
    :return:
    """
    print('convert timedeltas')
    df_sessions = pd.read_pickle(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle')

    # df_sessions.dropna(subset=['f_session_length'], inplace=True)
    # df_sessions.reset_index(drop=True, inplace=True)
    #
    # def maybe_convert(x):
    #     try:
    #         return round(x.total_seconds())
    #     except AttributeError:
    #         return x
    #
    #
    # df_sessions['f_session_length'] = df_sessions['f_session_length'].apply(maybe_convert)

    # overwrite f_session_length because it has errors (some values in milliseconds, some in seconds)
    df_sessions['f_session_length'] = df_sessions['session_length'].apply(lambda t : t.total_seconds())

    with open(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def convert_timedeleta(value):
    """
    Convert a time delta value to seconds
    :return:
    """
    print('convert timedeltas')
    # Round of the Milliseconds value
    return round(value.total_seconds() * 1000)
    # Round of the Milliseconds value


def filter_users():
    """
    Filter the sessions of user to only include participants that completed both questionnaires
    :return:
    """
    print('filter users')
    df_sessions = pd.read_pickle(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle')
    df_MRH1_raw = pd.read_csv(f'{questionnaires_dir}\MRH1.csv', sep=',')
    df_MRH2 = pd.read_csv(f'{questionnaires_dir}\MRH2.csv', sep=',')

    studIDs_mrh2 = df_MRH2['IM01_01'].apply(lambda x: x.upper())

    # Find the users to drop
    # to_drop = []
    # grouped_logs = df_sessions.groupby(['studyID'])
    # # Iterate over sessions
    # for name, df_group in grouped_logs:
    #     print(name)
    #     if name.upper() not in studIDs_mrh2.values:
    #         to_drop.append(name)

    to_drop = ['AN09BI', 'NI23HE', 'PI02MA']

    print(to_drop)
    for study_id in to_drop:
        df_sessions.drop(df_sessions.index[df_sessions['studyID'] == study_id], inplace=True)

    # df_sessions.to_csv(fr'{dataframe_dir_ml}\sessions_filtered_users.csv')
    with open(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def drop_sequences():
    """
    Drop the sequences features
    :return:
    """
    print('drop sequences')
    df_sessions = pd.read_pickle(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle')
    df_sessions.drop(columns=['f_sequences'], inplace=True)
    # df_sessions.to_csv(fr'{dataframe_dir_users}\user-sessions_features_all_noseq.csv')
    with open(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def reduce_feature_dimension():
    """
    Reduce the feature dimension by summarizing app features that only occurred under a threshold value in a feature_other
    :return:
    """
    print("reduce_feature_dimension")
    # Create colum with other for each type
    # Get all columns starting with feature id
    # Count occurences of 1
    # if under threshhold (5?) add to other categorie

    # f_clicks_app_category
    # f_scrolls_app_category_
    # f_scrolls_ package
    # f_clicks__packge
    # f_app_count
    # f_app_time
    # f_app_category_count_
    # f_app_category_time

    # repeat for all types
    threshold = 5
    feature_to_reduce = ['f_clicks_app_category', 'f_scrolls_app_category', 'f_scrolls', 'f_clicks', 'f_app_count', 'f_app_time', 'f_app_category_count', 'f_app_category_time']

    df_sessions = pd.read_pickle(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle')
    df_sessions.reset_index(drop=True, inplace=True)
    # df_sessions.fillna(0.0, inplace=True)
    # df_sessions.drop(columns = ['session_length', 'timestamp_1', 'timestamp_2', 'f_sequences_apps'])

    for item in feature_to_reduce:

        colums_f = [x for x in df_sessions.columns.values if x.startswith(item)]
        other_colum_name = f'{item}_other'
        print(other_colum_name)
        df_sessions[other_colum_name] = 0

        for column_name in colums_f:
            df_sessions[column_name] = df_sessions[column_name].fillna(0)

            count = (df_sessions[column_name] > 0).sum()

            if count <= threshold:
                df_sessions[other_colum_name] = df_sessions[other_colum_name] + df_sessions[column_name]
                df_sessions.drop(columns=[column_name], inplace=True)

                # old = df_sessions.loc[df_sessions[column] > 0, other_colum_name]
                # print(df_sessions[df_sessions[column] > 0, other_colum_name])
                # print(old)
                # dfs = df_sessions[df_sessions[column_name] > 0]
                # for index, row in enumerate(df_sessions[df_sessions[column_name] > 0].itertuples()):
                #     #print(row, type(row))
                #     #print(row[column_name][0], row.other_colum_name)
                #     #print( type(row[column]), row[column])
                #     #new = row[column_name] + row[other_colum_name]
                #     print(index)
                #     print(df_sessions.loc[index, column_name], df_sessions.loc[index, other_colum_name])
                #     pre = df_sessions.loc[index, other_colum_name] + df_sessions.loc[index, column_name]
                #     df_sessions.loc[index, other_colum_name] = pre
                #     print("new", df_sessions.loc[index, other_colum_name])

                # df_sessions.drop(columns=[column], inplace=True)

                # print(df_sessions[df_sessions[column] > 0, other_colum_name])
                # df_sessions.at[[df_sessions[column] == 1], other_colum_name] += count
                # print(t)
                # print(t.columns)
                # df_sessions.drop(columns=['f_sequences'], inplace=True)

    df_sessions.to_csv(fr'{dataframe_dir_ml}\user-sessions_features_all_f_reduction.csv')
    with open(fr'{dataframe_dir_ml}\user-sessions_features_all_f_reduction.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def concat_sessions():
    print('concat sessions')
    path_list = pathlib.Path(dataframe_dir_sessions_features).glob('**/*.pickle')
    sessions = []

    for data_path in path_list:
        path_in_str = str(data_path)
        sessions.append(pd.read_pickle(path_in_str))

    df_concat = pd.concat(sessions, ignore_index=False)
    df_concat.reset_index(drop=True, inplace=True)

    df_concat.to_csv(fr'{dataframe_dir_ml}\user-sessions_features_all.csv')
    with open(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle', 'wb') as f:
        pickle.dump(df_concat, f, pickle.HIGHEST_PROTOCOL)


def concat_features_dic():
    print('----------Concat session and feature list----------')
    path = fr'{dataframe_dir}\user-sessions_features.pickle'
    df_all_sessions = pd.read_pickle(path)
    all = []

    for key in df_all_sessions.keys():
        all.append(df_all_sessions[key])

    df_concat = pd.concat(all, ignore_index=False)

    df_concat.to_csv(fr'{dataframe_dir}\user-sessions_features_all.csv')
    with open(fr'{dataframe_dir}\user-sessions_features_all.pickle', 'wb') as f:
        pickle.dump(df_concat, f, pickle.HIGHEST_PROTOCOL)


def remove_personalised_features():
    """
    Remove the personalized feature to prepare the session for machine learning
    :return:
    """
    path = fr'{dataframe_dir_ml_labeled}\sessions_features_labeled_more_than_intention_with_esm.pickle'
    df_sessions = pd.read_pickle(path)

    df_sessions.drop(df_sessions.index[df_sessions['f_session_length'].isnull()], inplace=True)

    df_sessions = drop_esm_features(df_sessions)

    colums_age = [x for x in df_sessions.columns.values if x.startswith('f_demographics_age')]
    df_sessions.drop(columns=colums_age, inplace=True)

    colums_gender = [x for x in df_sessions.columns.values if x.startswith('f_demographics_gender')]
    df_sessions.drop(columns=colums_gender, inplace=True)

    colums_wlan_name = [x for x in df_sessions.columns.values if x.startswith('f_internet_connected_WIFI_')]
    df_sessions.drop(columns=colums_wlan_name, inplace=True)

    df_sessions.drop(columns=['f_absentminded_use', 'f_general_use'], inplace=True)

    df_sessions.to_csv(fr'{dataframe_dir_ml_labeled}\user-sessions_features_all_labled_more_than_intention_no_personalized.csv')

    with open(fr'{dataframe_dir_ml_labeled}\user-sessions_features_all_labled_more_than_intention_normal_age_no_esm_no_personal.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def clean_df():
    """
    Undo the age encoding into bags
    :return:
    """
    path = fr'{dataframe_dir_ml}\user-sessions_features_all.pickle'
    df_sessions = pd.read_pickle(path)

    path_questionnaire_1 = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData'
    df_MRH1 = pd.read_csv(f'{path_questionnaire_1}\MRH1.csv', sep=',')

    df_sessions.drop(df_sessions.index[df_sessions['f_session_length'].isnull()], inplace=True)

    colums_age = [x for x in df_sessions.columns.values if x.startswith('f_demographics_age_')]
    df_sessions.drop(columns=colums_age, inplace=True)

    # Add age instead of age groupds
    for index, session in enumerate(df_sessions.itertuples(index=False)):
        if not df_MRH1[df_MRH1['IM01_01'].values == session[2]].empty:
            df_qu_user = df_MRH1[df_MRH1['IM01_01'].values == session[2]].index.item()
            df_sessions.at[index, 'f_demographics_age'] = df_MRH1.loc[df_qu_user, 'SD02_01']


def calculuate_across_session_features():
    feature_path =  fr'{dataframe_dir_ml}\user-sessions_features_all_of.pickle'
    df_sessions = pd.read_pickle(feature_path)

    # --- over-session features ---

    # session length normalized
    features_to_normalize = ['f_session_length',
                             'f_time_since_last_session',
                             'f_glances_since_last_session',
                             'f_app_category_time_Communication',
                             'f_app_category_time_Finance',
                             'f_app_category_time_Internet',
                             'f_app_category_time_System',
                             'f_app_category_time_UNKNOWN',
                             'f_app_category_time_Social_Media',
                             'f_app_category_time_Health',
                             'f_app_category_time_Settings',
                             'f_app_category_time_Shopping',
                             'f_app_category_time_Audio_Entertainment',
                             'f_app_category_time_Photo',
                             'f_app_category_time_Orientation',
                             'f_app_category_time_Tools',
                             'f_app_category_time_Visual_Entertainment',
                             'f_app_category_time_Transportation',
                             'f_app_category_time_Gaming',
                             'f_app_category_time_Food',
                             'f_app_category_time_Security',  'f_app_category_time_Time',
                             'f_app_category_time_Knowledge',
                             'f_app_category_time_News', 'f_app_category_time_Weather',
                             'f_app_category_time_Career', 'f_app_category_time_Dating',
                             'f_app_category_time_Reading'
                             ]
    for featurename in features_to_normalize:
        usermeans = df_sessions.groupby('studyID')[featurename].mean()
        df_sessions[featurename+"_nom"] = df_sessions.apply(lambda row : row[featurename]/usermeans[row['studyID']], axis=1)

  #  df_sessions = df_sessions['f_session_length'] / df_sessions["f_session_length_nom"]


    with open(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    # # 1. Extract the logs from the raw database files
    # extractData()
    #
    # # 2. Extract the users without logs from raw database files
    # extractUser()
    #
    # # 3. Run the preprocessing steps like transform timestams and extract sessions usw
    # preprocessing()
    #
    # # 4. Extract the features from the logs and saves it to the sessions df
   # extract_features()
    #
    # # 5. concat all session and features df from each user to one
    # # concat_features_dic() #old
#    concat_sessions()

    # 6. Create the bag of apps for each sessions (using all session df)
#    bag_of_apps_create_vocab()
#    bag_of_apps_create_bags()

    # 7. Convert timedeltas to milliseconds and drop unused columns
# #    drop_sequences()
#     convert_timedeletas()
#
#     # Filter the session to give an overview over sessions with esm
#     ## filter_sessions_esm_user_based()
#
#     # 10. On hot encode colums like esm
#     ## one_hot_encoding_dummies()
#    # one_hot_encoding_scilearn()
#
#     # Bag and endcode demograpgics age
#     ## demographics_encode_age()

    # 11. Filter outliners
    filter_sessions_outliners_all()

    # 11.1
    calculuate_across_session_features()
    #
    # # 12. Only use users that completed the second questionnaire
    # filter_users()
    #
    # # 13. reduce feautre dimension by grouping columns together
    # reduce_feature_dimension()
    #
    # # 13. create labels as targets (only works with onhot encoded data)
    # create_labels_single()
    # # labeling_combined()
    #
    # # 14. If needed - remove personal features like age, gender or absentminded/general use scores
    # remove_personalised_features()

    # clean_df()
