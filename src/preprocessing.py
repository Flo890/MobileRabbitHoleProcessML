import pandas as pd
import pathlib
import pickle
import getFromJson
import prepareTimestamps
import extractSessions
import featureExtraction

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
    Run all necessary preprocessing steps
    """
    path_list = pathlib.Path(dataframe_dir_live_logs_sorted).glob('**/*.pickle')

    dic_device = {}
    dic_installed_apps = {}

    user_sessions = {}
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

        with open(fr'{dataframe_dir_live}\{data_path.stem}.pickle', 'wb') as f:
            pickle.dump(extracted[0], f, pickle.HIGHEST_PROTOCOL)
            f.close()

    with open(fr'{dataframe_dir_live}\user-sessions.pickle', 'wb') as f:
        pickle.dump(user_sessions, f, pickle.HIGHEST_PROTOCOL)

    with open(fr'{dataframe_dir_live}\user-device_info.pickle', 'wb') as f:
        pickle.dump(dic_device, f, pickle.HIGHEST_PROTOCOL)

    with open(fr'{dataframe_dir_live}\user-installed_apps.pickle', 'wb') as f:
        pickle.dump(dic_installed_apps, f, pickle.HIGHEST_PROTOCOL)


def extract_features():
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


def print_test_df():
    print('print test')
    path_testfile = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\usersorted\SO23BA.pickle'
    t = pd.read_pickle(path_testfile)
    # time = t.correct_timestamp
    t.to_csv(fr'{dataframe_dir_test}\SO23BA_logs_new.csv')

    path_testfile_sessions = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\user-sessions.pickle'
    pickle_in = open(path_testfile_sessions, "rb")
    sessions = pickle.load(pickle_in)
    test = sessions['SO23BA']
    test.to_csv(fr'{dataframe_dir_test}\SO23BA_sessions_new.csv')


if __name__ == '__main__':
    # extractData()
    # preprocessing()
    extract_features()

    print_test_df()
