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
import datetime

# Get rid of duplicated
# Remove overhead logs aka TYPE_WINDOW_CONTENT_CHANGED
# extract installed apps and dive info -> might add to user df
# get rid of RabbitHole events aka admin and packagename com.lmu.rabbitholetracker
# Clean Timestamps
# Get sessions
# add events that were not in sessions? like sms, calls or notifications?

raw_data_dir_test = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\t'
raw_data_user = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\live\recent2022-04-17T21 44 59Z_absentmindedtrack-default-rtdb_data.json.gz'
raw_data_live = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\live'
logs_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\logFiles'
user_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\cleanedUsers'
dataframe_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes'
dataframe_dir_test = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\testing'
dataframe_dir_live = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\live'

dataframe_dir_users = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\users'
dataframe_dir_sessions = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\sessions'
dataframe_dir_sessions_features = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\sessions_features'
dataframe_dir_sessions_filtered = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\sessions_filtered'
dataframe_dir_bagofapps_vocab = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\bag_of_apps_vocabs'

dataframe_dir_live_logs_sorted = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\usersorted_logs'
dataframe_dir_live_logs_sorted_preprocessed = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\usersorted_logs_preprocessed'

path_testfile_sessions = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\user-sessions.pickle'
path_testfile_sessions_all = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\user_sessions_all.pickle'
clean_users_dir_path = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\users'
clean_users_path = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\users\cleaned_users.pickle'


def extractData():
    # only run once
    # getFromJson.extractUsers(file_path=raw_data_live, end_directory=dataframe_dir_live, is_gzip=True)

    # extract logs for files
    # getFromJson.extract_logs(directory=raw_data_live, end_directory=dataframe_dir_live, save_type=3, is_gzip=True)

    # concat all extracted logs
    # getFromJson.concat_user_logs(logs_dic_directory=dataframe_dir_live, end_directory=dataframe_dir_live_logs_sorted)
    # user_list = getFromJson.getStudIDlist(clean_users_path)
    # user_list = ['IN24RA', 'so30vo', 'NI23HE', 'vi03th', 'AN23GE', 'BI21FR', 'ju05ab', 'EW11FR', 'KE18ZA', 'ZA23PA', 'TH28RU', 'ha07fl', 'MO22GO', 'NA07WE', 'AN09BI', 'PI02MA', 'EL28LO', 'BR04WO', 'IR03JO', 'PA29BU', 'SHTUSI', 'IM12WE', 'AY1221', 'ta22ar', 'CR11AI', 'MA06DR', 'AN06PE', 'SA23BR', 'CO07FA', 'IR18RA', 'BA10SC', 'SO23BA', 'LE13FO']
    user_list = [('LE13FO', 'vi03th'), ('AN23GE', 'BI21FR'), ('ju05ab', 'EW11FR'), ('KE18ZA', 'ZA23PA'), ('TH28RU', 'ha07fl'), ('MO22GO', 'NA07WE'), ('AN09BI', 'PI02MA'), ('EL28LO', 'BR04WO'),
                 ('IR03JO', 'PA29BU'), ('SHTUSI', 'IM12WE'), ('AY1221', 'ta22ar'), ('CR11AI', 'MA06DR'), ('AN06PE', 'SA23BR'), ('CO07FA', 'IR18RA'), ('BA10SC', 'SO23BA')]
    user_list = [('LE13FO', 'vi03th', 'AN23GE', 'BI21FR'), ('ju05ab', 'EW11FR', 'KE18ZA', 'ZA23PA'), ('TH28RU', 'ha07fl', 'MO22GO', 'NA07WE'), ('AN09BI', 'PI02MA', 'EL28LO', 'BR04WO'),
                 ('IR03JO', 'PA29BU', 'SHTUSI', 'IM12WE'), ('AY1221', 'ta22ar', 'CR11AI', 'MA06DR'), ('AN06PE', 'SA23BR', 'CO07FA', 'IR18RA'), ('BA10SC', 'SO23BA', 'tt', 'zz')]

    for item in user_list:
        print(item)
        getFromJson.concat_logs_one_user(logs_dic_directory=dataframe_dir_live, end_directory=dataframe_dir_live_logs_sorted, studyID=item)


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

    # Get rid of duplicated   unhashable type: 'dict'
    # df_logs.drop_duplicates(inplace=True) # (subset=['brand', 'style'], keep='last')

    # TODO clean notifications

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
    path_list = pathlib.Path(dataframe_dir_live_logs_sorted_preprocessed).glob('**/*.pickle')
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

            with open(fr'{dataframe_dir_bagofapps_vocab}\{data_path.stem}-vocab', 'wb') as f:
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
            # plt.show()
            # (df_session_features['f_session_length'] / pd.Timedelta(seconds=1)).hist()
            # plt.show()

            # sns.distplot(df_session_features['f_session_length'] / pd.Timedelta(milliseconds=1))
            # plt.show()
            # sns.boxplot(df_session_features['f_session_length'] / pd.Timedelta(milliseconds=1))
            # plt.show()

            upper_limit = df_session_features['f_session_length'].quantile(0.99)
            lower_limit = df_session_features['f_session_length'].quantile(0.01)

            # new_df = df_session_features[(df_session_features['f_session_length'] <= upper_limit) & (df_session_features['f_session_length'] >= lower_limit)]
            # print(type(new_df))
            # sns.boxplot( new_df['f_session_length'] / pd.Timedelta(milliseconds=1))
            # plt.show()

            print(upper_limit)
            print(lower_limit)

            # new_df = df[(df['Height'] <= 74.78) & (df['Height'] >= 58.13)]

            # drop sessions
            # df_session_features = df_session_features.drop(df_session_features['f_session_length'] > threshold_max_cap)
            df_filtered_esm = df_session_features[(df_session_features['f_session_length'] > threshold) & (df_session_features['f_esm_finished_intention'] != '')]

            with open(fr'{dataframe_dir_sessions_filtered}\{data_path.stem}-sessions_filtered.pickle', 'wb') as f:
                pickle.dump(df_filtered_esm, f, pickle.HIGHEST_PROTOCOL)

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

            session_stats.append({'studyID': data_path.stem, 'session_count': len(df_session_features), 'esm_over45s_count': len(df_filtered_esm), 'session_length_mean': session_length_mean,
                                  'sessions_not_finished': len(df_esm_rabbithole_finished),
                                  'session_more_than_intention': len(df_esm_rabbithole_more), 'sessions_not_finished_and_more': len(df_esm_rabbithole_finished_more)})

    # Calculate mean session count
    # Calculate mean session lenght (for each user?, all)
    # Calculate mean rabbit hole sessions
    sessions_list = pd.DataFrame(session_stats)
    sessions_list.to_csv(fr'{user_dir}\sessions_stats.csv')


def filter_sessions_outliners_all():
    # Filter the relevant sessions
    # Get all sessions that are over 45s and have a esm answers (f_esm_finished_intention not empty)
    threshold = pd.Timedelta(seconds=45)
    threshold_max_cap = pd.Timedelta(seconds=25200)  # 7 stunden
    feature_path = fr'{dataframe_dir}\user-sessions_features_all.pickle'
    df_all_sessions = pd.read_pickle(feature_path)

    session_stats = []
    df_sessions_filtered = {}

    # sns.distplot(df_session_features['f_session_length'] / pd.Timedelta(milliseconds=1))
    # plt.show()
    sns.boxplot(df_all_sessions['f_session_length'] / pd.Timedelta(milliseconds=1))
    plt.show()

    upper_limit = df_all_sessions['f_session_length'].quantile(0.996)  # bis 5.5h bei 0.995 bis 3.5h
    lower_limit = df_all_sessions['f_session_length'].quantile(0.01)

    new_df = df_all_sessions[(df_all_sessions['f_session_length'] <= upper_limit) & (df_all_sessions['f_session_length'] >= lower_limit)]
    print(type(new_df))
    sns.boxplot(new_df['f_session_length'] / pd.Timedelta(milliseconds=1))
    plt.show()

    print(upper_limit)
    print(lower_limit)

    # # drop sessions
    # # df_session_features = df_session_features.drop(df_session_features['f_session_length'] > threshold_max_cap)
    # df_filtered_esm = df_all_sessions[(df_all_sessions['f_session_length'] > threshold) & (df_all_sessions['f_esm_finished_intention'] != '')]
    #
    # # TODO Change yes to and colums?
    # df_esm_rabbithole_finished = df_filtered_esm[(df_filtered_esm['f_esm_finished_intention'] == 'No')]
    # df_esm_rabbithole_more = df_filtered_esm[(df_filtered_esm['f_esm_more_than_intention'] == 'Yes')]
    # df_esm_rabbithole_finished_more = df_filtered_esm[(df_filtered_esm['f_esm_finished_intention'] == 'No') & (df_filtered_esm['f_esm_more_than_intention'] != 'Yes')]
    #
    # session_length_mean = pd.to_timedelta(df_all_sessions['f_session_length']).mean()
    # # print('sessionlength mean', session_length_mean)

    return df_sessions_filtered



def one_hot_encoding():
    print('on hot encoding')
    df_sessions = pd.read_pickle(fr'{dataframe_dir_users}\user-sessions_features_all.pickle')
    # df_sessions.to_csv(fr'{dataframe_dir_users}\user-sessions_features_all.csv')
    df_sessions = df_sessions.replace(r'^\s*$', np.nan, regex=True)
    to_encode = ['f_demographics_gender', 'f_esm_finished_intention', 'f_esm_more_than_intention', 'f_esm_emotion', 'f_esm_track_of_time', 'f_esm_track_of_space', 'f_esm_regret', 'f_esm_agency',
                 'f_hour_of_day', 'f_weekday']

    for column in to_encode:
        df_encoded = pd.get_dummies(df_sessions[column], prefix=column)
        df_sessions.drop(columns=[column], inplace=True)
        df_sessions = pd.concat([df_sessions, df_encoded], axis=1)

    with open(fr'{dataframe_dir_users}\user-sessions_features_1hotencoded', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def bag_of_apps_create_vocab():
    print('create bag of apps vocabulary')
    # get the vocabulary, which is every application package used by every user?
    # create vocab from every users app list
    path_list = pathlib.Path(dataframe_dir_bagofapps_vocab).glob('**/*.pickle')
    vocab = []
    for data_path in path_list:
        path_in_str = str(data_path)
        user_vocab = pd.read_pickle(path_in_str)
        vocab = list(set(vocab + user_vocab))
        print(vocab)

    with open(fr'{dataframe_dir_bagofapps_vocab}\bag_of_apps_vocab', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


def bag_of_apps_create_bags():
    """
    Create a bag of apps vector for each session in the df of all sessions
    :return:
    """
    print('create bag of apps')
    # go over every session and create vector of app count
    vocab = pd.read_pickle(fr'{dataframe_dir_bagofapps_vocab}\bag_of_apps_vocab')
    df_sessions = pd.read_pickle(fr'{dataframe_dir_users}\user-sessions_features_all.pickle')
    df_sessions['f_bag_of_apps'] = np.zeros(len(vocab))

    for session in df_sessions.itertuples():
        # sessions['f_sequences_apps']
        print(session.index)

        app_list = session.f_sequences_apps[0][0]
        bag_vector = np.zeros(len(vocab))
        for a in app_list:
            for i, app in enumerate(vocab):
                if app == a:
                    bag_vector[i] += 1

        df_sessions.loc[session.index, 'f_bag_of_apps'] = bag_vector

    with open(fr'{dataframe_dir_users}\user-sessions_features_all.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)

def convert_timedeletas():
    print('convert timedeltas')
    timestamp_1 = datetime.datetime(2019, 6, 3, 8, 12, 16, 104000)
    timestamp_2 = datetime.datetime(2019, 6, 3, 8, 12, 25, 249000)
    # # Get different between two datetime as timedelta object.
    diff = (timestamp_2 - timestamp_1)
    print(diff)
    print(diff.total_seconds() * 1000)
    return round(diff.total_seconds() * 1000)
    # Round of the Milliseconds value
    #diff_in_milliseconds = round(diff_in_milliseconds)


def filter_users():
    print('filter users')
    # TODO

def concat_sessions():
    path_list = pathlib.Path(dataframe_dir_sessions_features).glob('**/*.pickle')
    sessions = []

    for data_path in path_list:
        path_in_str = str(data_path)
        sessions.append(pd.read_pickle(path_in_str))

    df_concat = pd.concat(sessions, ignore_index=False)

    pd.concat(sessions, axis=0, ignore_index=True)
    with open(fr'{dataframe_dir_users}\user-sessions_features_all.pickle', 'wb') as f:
        pickle.dump(df_concat, f, pickle.HIGHEST_PROTOCOL)


def concat_features_dic():
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
    # path_testfile = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\usersorted_logs_preprocessed\BR04WO.pickle'
    # t = pd.read_pickle(path_testfile)
    # # time = t.correct_timestamp
    # t.to_csv(fr'{user_dir}\BR04WO_logs_preprocesses.csv')

    path_testfile = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\sessions_features\CO07Fa-sessions_features.pickle'
    t = pd.read_pickle(path_testfile)
    # time = t.correct_timestamp
    # get sequencelsit: t.f_sequences[0][0]

    # t.to_csv(fr'{dataframe_dir_users}\CO07Fa_sessions-features.csv')

    # path_testfile = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\sessions_filtered\CO07Fa-sessions_features-sessions_filtered.pickle'
    # t = pd.read_pickle(path_testfile)
    # # time = t.correct_timestamp
    # t.to_csv(fr'{dataframe_dir_users}\CO07Fa_sessions-filtered.csv')

    # path_testfile_sessions = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\user-sessions_features.pickle'
    # pickle_in = open(path_testfile_sessions, "rb")
    # sessions = pickle.load(pickle_in)
    # test = sessions['SO23BA']
    # test.to_csv(fr'{dataframe_dir_test}\SO23BA_sessions.csv')

    # path_testfile = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\users\user-device_info.pickle'
    # device = pd.read_pickle(path_testfile)
    # device.to_csv(fr'{user_dir}\user-device_info.csv')
    #
    # path_testfile = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\users\user-installed_apps.pickle'
    # installed = pd.read_pickle(path_testfile)
    # installed.to_csv(fr'{user_dir}\user-installed_apps.csv')


def test():
    path_testfile = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\usersorted_logs_preprocessed\CO07Fa.pickle'
    t = pd.read_pickle(path_testfile)
    df = t[(t['eventName'].values == 'ESM')]
    df.to_csv(fr'{dataframe_dir_users}\CO07Fatest.csv')
    print(df[['eventName', 'event']])


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    # extractData()
    # extractUser()
    # check_for_activity()
    # preprocessing()
    # extract_features()
    # concat_features_dic() #old
    concat_sessions()

    # filter_sessions_esm_user_based()

    #one_hot_encoding()
    #result = convert_timedeletas()
    #print(result)

    # path_testfile_sessions = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\example_sessions_features.csv'

    # path_testfile_sessions = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\user-sessions_features.pickle'
    # pickle_in = open(path_testfile_sessions, "rb")
    # sessions = pickle.load(pickle_in)
    # test = sessions['SO23BA']
    # # test = pd.read_csv(path_testfile_sessions)
    # # filter_sessions()
    # filter_sessions_all()

    # print_test_df()
    # test()
