import pandas as pd
import pathlib
import pickle
import getFromJson
import prepareTimestamps
import extractSessions

# Get rid of duplicated
# Remove overhead logs aka TYPE_WINDOW_CONTENT_CHANGED
# extract installed apps and dive info -> might add to user df
# get rid of RabbitHole events aka admin and packagename com.lmu.rabbitholetracker
# Clean Timestamps
# Get sessions
# add events that were not in sessions? like sms, calls or notifications?

raw_data_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\test'
raw_data_live = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\live'
logs_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\logFiles'
user_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\cleanedUsers'
dataframe_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\testing'

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

    # Get rid of duplicated     unhashable type: 'dict'
    # df_logs.drop_duplicates(inplace=True) # (subset=['brand', 'style'], keep='last')

    # extract installed apps and dive info -> might add to user df

    return df_logs


def extractData():
    # only run once
    # getFromJson.extractUsers(file_path=raw_data_user, end_directory=user_dir, is_gzip=True)

    getFromJson.extract_logs(directory=raw_data_live, end_directory=dataframe_dir, save_type=3, is_gzip=True)


def preprocessing():
    """
    Run all necessary preprocessing steps
    """
    path_list = pathlib.Path(dataframe_dir).glob('**/*.pickle')

    user_sessions = {}
    for data_path in path_list:
        path_in_str = str(data_path)
        df_logs = pd.read_pickle(path_in_str)
        print(f'preprocess {data_path.stem}')

        df_logs = cleanup(df_logs)
        df_logs = prepareTimestamps.process_timestamps(df_logs)
        # extract Metadata
        df_logs = getFromJson.extractMetaData(df_logs)

        extracted = extractSessions.extract_sessions(df_logs)
        user_sessions[data_path.stem] = extracted[1]

        with open(fr'{dataframe_dir}\{data_path.stem}.pickle', 'wb') as f:
            pickle.dump(extracted[0], f, pickle.HIGHEST_PROTOCOL)
            f.close()

    with open(fr'{dataframe_dir}\user-sessions.pickle', 'wb') as f:
        pickle.dump(user_sessions, f, pickle.HIGHEST_PROTOCOL)


def print_test_df():
    path_testfile = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\testing\SO23BA.pickle'
    t = pd.read_pickle(path_testfile)
    t.to_csv(fr'{dataframe_dir}\SO23BAtest_logs.csv')

    path_testfile_sessions = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\user-sessions.pickle'
    pickle_in = open(path_testfile_sessions,"rb")
    sessions = pickle.load(pickle_in)
    test = sessions['SO23BA']
    test.to_csv(fr'{dataframe_dir}\SO23BAtest_sessions.csv')




if __name__ == '__main__':

    # extractData()
    # cleanup()
    # preprocessing()

    print_test_df()
