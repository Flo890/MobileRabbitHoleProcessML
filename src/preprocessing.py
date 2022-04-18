import pandas as pd
import pathlib
import pickle
import getFromJson
import prepareTimestamps
import extractSessions
import featureExtraction
import matplotlib.pyplot as plt
import seaborn as sns

# Get rid of duplicated
# Remove overhead logs aka TYPE_WINDOW_CONTENT_CHANGED
# extract installed apps and dive info -> might add to user df
# get rid of RabbitHole events aka admin and packagename com.lmu.rabbitholetracker
# Clean Timestamps
# Get sessions
# add events that were not in sessions? like sms, calls or notifications?

raw_data_dir_test = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\t'
raw_data_user = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\live\recent_2022-03-26T23 31 00Z_absentmindedtrack-default-rtdb_data.json.gz'
raw_data_live = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\live'
logs_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\logFiles'
user_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\cleanedUsers'
dataframe_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes'
dataframe_dir_test = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\testing'
dataframe_dir_live = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\live'
dataframe_dir_live_logs_sorted = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\usersorted'
path_testfile_sessions = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\user-sessions.pickle'
path_testfile_sessions_all = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\user_sessions_all.pickle'


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

    # Get rid of duplicated   unhashable type: 'dict'
    # df_logs.drop_duplicates(inplace=True) # (subset=['brand', 'style'], keep='last')

    # TODO clean notifications

    return df_logs


def extractData():
    # only run once
    # getFromJson.extractUsers(file_path=raw_data_live, end_directory=dataframe_dir_live, is_gzip=True)

    # extract logs for files
    getFromJson.extract_logs(directory=raw_data_dir_test, end_directory=dataframe_dir_test, save_type=3, is_gzip=True)
    # concat all extracted logs
    getFromJson.concat_user_logs(logs_dic_directory=dataframe_dir_test, end_directory=dataframe_dir_live_logs_sorted)


def preprocessing():
    """
    Run all necessary preprocessing steps,
     prepare timestamps, flattens metadata, extract user sessions, device info and installed apps
    """
    path_list = pathlib.Path(dataframe_dir_live_logs_sorted).glob('**/*.pickle')

    dic_device = {}
    dic_installed_apps = {}

    user_sessions = {}
    list_user_sessions_all = []

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
            dic_device[data_path.stem] = pd.DataFrame([{'model': info.event, 'releaseVersion': info.description, 'sdkVersion': info.name, 'manufacturer': info.packageName}])

        # LogEventName.DEVICE_INFO, timestamp, event= model, description = releaseVersion, name = sdkVersion.toString(), packageName = manufacturer )
        # INSTALLED_APP
        dic_installed_apps[data_path.stem] = df_logs[df_logs['event'] == 'INSTALLED_APP']['packageName']

        extracted = extractSessions.extract_sessions(df_logs)
        user_sessions[data_path.stem] = extracted[1]
        list_user_sessions_all.append(extracted[1])

        with open(fr'{dataframe_dir_live}\{data_path.stem}.pickle', 'wb') as f:
            pickle.dump(extracted[0], f, pickle.HIGHEST_PROTOCOL)
            f.close()

    with open(fr'{dataframe_dir_live}\user-sessions.pickle', 'wb') as f:
        pickle.dump(user_sessions, f, pickle.HIGHEST_PROTOCOL)

    # user_sessions_all = pd.concat(list_user_sessions_all, ignore_index=False)
    # user_sessions_all.to_csv(fr'{dataframe_dir_live}\user_sessions_all.csv')
    # with open(fr'{dataframe_dir_live}\user_sessions_all.pickle', 'wb') as f:
    #     pickle.dump(user_sessions, f, pickle.HIGHEST_PROTOCOL)

    with open(fr'{dataframe_dir_live}\user-device_info.pickle', 'wb') as f:
        pickle.dump(dic_device, f, pickle.HIGHEST_PROTOCOL)

    with open(fr'{dataframe_dir_live}\user-installed_apps.pickle', 'wb') as f:
        pickle.dump(dic_installed_apps, f, pickle.HIGHEST_PROTOCOL)


def extract_features():
    """
    Extract the features for each user from the log file and add it to the sessions and feature dataframe
    :return: saves the session list to file
    """
    path_list = pathlib.Path(dataframe_dir_live).glob('**/*.pickle')
    df_all_sessions = pd.read_pickle(path_testfile_sessions)
    # print(df_all_sessions)

    for data_path in path_list:
        path_in_str = str(data_path)
        # df_all_logs = pd.read_pickle(path_in_str)

        print(data_path.stem)
        # session_for_id = df_all_sessions[data_path.stem]

        if not df_all_sessions[data_path.stem].empty:
            print("not empty")
            df_session_features = featureExtraction.get_features_for_session(df_logs=pd.read_pickle(path_in_str), df_sessions=df_all_sessions[data_path.stem])
            df_all_sessions[data_path.stem] = df_session_features

    test = df_all_sessions['SO23BA']
    test.to_csv(fr'{dataframe_dir_test}\SO23BA_sessions_features.csv')
    with open(fr'{dataframe_dir}\user-sessions_features.pickle', 'wb') as f:
        pickle.dump(df_all_sessions, f, pickle.HIGHEST_PROTOCOL)


def filter_sessions_user_based():
    """
    Filter the sessions with ESM and above 45s, calculate and save stats, remove outliners
    :return:
    """
    # Filter the relevant sessions
    # Get all sessions that are over 45s and have a esm answers (f_esm_finished_intention not empty)
    threshold = pd.Timedelta(seconds=45)
    threshold_max_cap = pd.Timedelta(seconds=25200)  # 7 stunden
    feature_path = fr'{dataframe_dir}\user-sessions_features.pickle'
    df_all_sessions = pd.read_pickle(feature_path)

    session_stats = []
    df_sessions_filtered = {}
    sessionlength = []

    for studyID in df_all_sessions:
        print(studyID)
        if not df_all_sessions[studyID].empty:
            df_session_features = df_all_sessions[studyID]
            print(df_session_features['f_session_length'].astype('timedelta64[s]').head(), len(df_session_features))
            # testtt = df_session_features['f_session_length'].values.astype('timedelta64[s]')

            # ax = sns.stripplot(x= (df_session_features['f_session_length'] / pd.Timedelta(seconds=1)))
            sessionlength.append(df_session_features['f_session_length'])
            print(len(df_session_features['f_session_length']))
            # plt.show()
            # (df_session_features['f_session_length'] / pd.Timedelta(seconds=1)).hist()
            # plt.show()

            #sns.distplot(df_session_features['f_session_length'] / pd.Timedelta(milliseconds=1))
            #plt.show()
            sns.boxplot(df_session_features['f_session_length'] / pd.Timedelta(milliseconds=1))
            plt.show()

            upper_limit = df_session_features['f_session_length'].quantile(0.99)
            lower_limit = df_session_features['f_session_length'].quantile(0.01)

            new_df = df_session_features[(df_session_features['f_session_length'] <= upper_limit) & (df_session_features['f_session_length'] >= lower_limit)]
            print(type(new_df))
            sns.boxplot( new_df['f_session_length'] / pd.Timedelta(milliseconds=1))
            plt.show()

            print(upper_limit)
            print(lower_limit)

            # new_df = df[(df['Height'] <= 74.78) & (df['Height'] >= 58.13)]

            # drop sessions
            # df_session_features = df_session_features.drop(df_session_features['f_session_length'] > threshold_max_cap)
            df_filtered_esm = df_session_features[(df_session_features['f_session_length'] > threshold) & (df_session_features['f_esm_finished_intention'] != '')]
            df_sessions_filtered[studyID] = df_filtered_esm

            # TODO Change yes to and colums?
            df_esm_rabbithole_finished = df_filtered_esm[(df_filtered_esm['f_esm_finished_intention'] == 'No')]
            df_esm_rabbithole_more = df_filtered_esm[(df_filtered_esm['f_esm_more_than_intention'] == 'Yes')]
            df_esm_rabbithole_finished_more = df_filtered_esm[(df_filtered_esm['f_esm_finished_intention'] == 'No') & (df_filtered_esm['f_esm_more_than_intention'] != 'Yes')]
            # print('not_finsihed', len(df_esm_rabbithole_finished))
            # print("more", len(df_esm_rabbithole_more))
            # print('finished more', len(df_esm_rabbithole_finished_more))

            # TODO save counts and % ?
            # print('sessioncount', len(df_session_features))
            # print('esm sessioncounts', len(df_filtered_esm))

            session_length_mean = pd.to_timedelta(df_session_features['f_session_length']).mean()
            # print('sessionlength mean', session_length_mean)

            session_stats.append({'studyID': studyID, 'session_count': len(df_session_features), 'esm_over45s_count': len(df_filtered_esm), 'session_length_mean': session_length_mean,
                                  'sessions_not_finished': len(df_esm_rabbithole_finished),
                                  'session_more_than_intention': len(df_esm_rabbithole_more), 'sessions_not_finished_and_more': len(df_esm_rabbithole_finished_more)})

    # Calculate mean session count
    # Calculate mean session lenght (for each user?, all)
    # Calculate mean rabbit hole sessions
    sessions_list = pd.DataFrame(session_stats)
    sessions_list.to_csv(fr'{dataframe_dir}\sessions_stats.csv')

    with open(fr'{dataframe_dir}\user-sessions_filtered.pickle', 'wb') as f:
        pickle.dump(df_sessions_filtered, f, pickle.HIGHEST_PROTOCOL)

    return df_sessions_filtered


def filter_sessions_all():
    # Filter the relevant sessions
    # Get all sessions that are over 45s and have a esm answers (f_esm_finished_intention not empty)
    threshold = pd.Timedelta(seconds=45)
    threshold_max_cap = pd.Timedelta(seconds=25200)  # 7 stunden
    feature_path = fr'{dataframe_dir}\user-sessions_features_all.pickle'
    df_all_sessions = pd.read_pickle(feature_path)

    session_stats = []
    df_sessions_filtered = {}


    #sns.distplot(df_session_features['f_session_length'] / pd.Timedelta(milliseconds=1))
    #plt.show()
    sns.boxplot(df_all_sessions['f_session_length'] / pd.Timedelta(milliseconds=1))
    plt.show()

    upper_limit = df_all_sessions['f_session_length'].quantile(0.996) #bis 5.5h bei 0.995 bis 3.5h
    lower_limit = df_all_sessions['f_session_length'].quantile(0.01)

    new_df = df_all_sessions[(df_all_sessions['f_session_length'] <= upper_limit) & (df_all_sessions['f_session_length'] >= lower_limit)]
    print(type(new_df))
    sns.boxplot( new_df['f_session_length'] / pd.Timedelta(milliseconds=1))
    plt.show()

    print(upper_limit)
    print(lower_limit)

    # drop sessions
    # df_session_features = df_session_features.drop(df_session_features['f_session_length'] > threshold_max_cap)
    df_filtered_esm = df_all_sessions[(df_all_sessions['f_session_length'] > threshold) & (df_all_sessions['f_esm_finished_intention'] != '')]

    # TODO Change yes to and colums?
    df_esm_rabbithole_finished = df_filtered_esm[(df_filtered_esm['f_esm_finished_intention'] == 'No')]
    df_esm_rabbithole_more = df_filtered_esm[(df_filtered_esm['f_esm_more_than_intention'] == 'Yes')]
    df_esm_rabbithole_finished_more = df_filtered_esm[(df_filtered_esm['f_esm_finished_intention'] == 'No') & (df_filtered_esm['f_esm_more_than_intention'] != 'Yes')]

    session_length_mean = pd.to_timedelta(df_all_sessions['f_session_length']).mean()
    # print('sessionlength mean', session_length_mean)


    return df_sessions_filtered


def concat_features():
    print('----------Concat session and feature list----------')
    path = fr'{dataframe_dir}\user-sessions_features.pickle'
    df_all_sessions = pd.read_pickle(path)
    all = []

    for key in df_all_sessions.keys():
        all.append(df_all_sessions[key])

    df_concat = pd.concat(all, ignore_index=False)

    df_concat.to_csv(fr'{dataframe_dir}\all_session_features.csv')
    with open(fr'{dataframe_dir}\user-sessions_features_all.pickle', 'wb') as f:
        pickle.dump(df_concat, f, pickle.HIGHEST_PROTOCOL)


def print_test_df():
    print('print test')
    path_testfile = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\usersorted\SO23BA.pickle'
    t = pd.read_pickle(path_testfile)
    # time = t.correct_timestamp
    t.to_csv(fr'{dataframe_dir_test}\SO23BA_logs_new.csv')

    path_testfile_sessions = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\user-sessions_features.pickle'
    pickle_in = open(path_testfile_sessions, "rb")
    sessions = pickle.load(pickle_in)
    test = sessions['SO23BA']
    test.to_csv(fr'{dataframe_dir_test}\SO23BA_sessions.csv')


if __name__ == '__main__':
    # extractData()
    # preprocessing()
    #extract_features()
    #concat_features()

    # path_testfile_sessions = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\example_sessions_features.csv'

    path_testfile_sessions = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\user-sessions_features.pickle'
    pickle_in = open(path_testfile_sessions, "rb")
    sessions = pickle.load(pickle_in)
    test = sessions['SO23BA']
    # test = pd.read_csv(path_testfile_sessions)
    # filter_sessions()
    filter_sessions_all()

    # print_test_df(test)
