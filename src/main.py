import datetime
import json
import pathlib
import re
import pandas as pd
import numpy as np
from cleanJson import CleanJson


def import_json(directory_files, dir_dataframe):
    """
    Import the json logs form one user into a dataframe and saves it as pickel.
    :param directory_files: the directory where the json log files of one user are located
    :param dir_dataframe: directoryy where to store the outputed dataframes as pickel
    """
    print("import json logs")
    pathlist = pathlib.Path(directory_files).glob('**/*.json')

    logs = []
    for data_path in pathlist:
        path_in_str = str(data_path)

        # with open(path_in_str, encoding='utf-8') as data:
        #     d = json.load(data)

        # read the json to dataFrame
        df = pd.read_json(path_in_str, orient="index")
        # print(df.head())

        logs.append(df)

        # TODO maybe delete all duplicate rows?
        # TODO extract metaData?
        # logsmeta = extractMetaData(logs)
        # logs = pd.read_pickle(file_name)

    logs_final = pd.concat(logs, ignore_index=False)

    print(logs_final.head())
    print(f"lenght {logs_final.size}")
    filename = pathlib.Path(directory_files).name
    logs_final.to_pickle(f'{dir_dataframe}\{filename}')


def extract_sessions(directory_files):
    """
    :param directory_files: the path were every user dataframe as pickel is stored
    """""
    print("extract sessions")
    # TODO for all user files
    # path_list = pathlib.Path(directory_files).glob('**/*.pkl')
    # for data_path in path_list:
    #     path_in_str = str(data_path)

    logs = pd.read_pickle(directory_files)

    # drop rows without an id
    # TODO logs.drop(logs.index[logs['id'] == ''], inplace=True)
    logs.drop(logs.index[logs['event'] == 'TYPE_WINDOW_CONTENT_CHANGED'], inplace=True)
    # TODO Clean notification duplicates?

    logs = logs.reset_index().set_index(logs['timestamp'], drop=False)
    print(logs.columns.values)

    # logs['timestamp'] = pd.to_datetime(logs['timestamp'], yearfirst=True, unit='ms')
    # TODO timezoneoffset

    # https://stackoverflow.com/a/56708633/16841176
    # logs['id'].replace('', np.nan, inplace=True)
    # logs.dropna(subset=['id'], inplace=True)

    session_grouped = logs.groupby(['id'])
    # print(session_grouped[['id', 'timestamp']].head(10))
    # groups = session_grouped.groups
    # print(session_grouped.head(1))
    session_grouped = session_grouped.apply(cut_sessions)

    # temp_session = logs[logs.duplicated(['id'])]


def discard_sessions(group):
    print("discard session")
    # TODO drop sessions that have under 10 values/ other criteria

def cut_sessions(group):
    print("cut sessions")
    # by_state.get_group("PA")
    # group.loc[: df[(df['Status'] == 'Open')].index[0], :]
    group_count = group.size
    screen = group.query('eventName  == "USAGE_EVENTS"')
    # group.query('`Sender email`.str.endswith("@shop.com")')
    # print(group_count)
    # print(screen)


    # slice at first occurrence of phone lock
    # slicetest = : group[(group.eventName == 'USAGE_EVENTS')] :
    # df.A.ne('a').idxmax()
    test = group.apply(lambda x: x[(x['event'].values == 'ACTIVITY_PAUSED').argmax():])
    print(test)
    # cut = group.loc[: group[(group['event'] == 'OFF_LOCKED')].index[0], :]



if __name__ == '__main__':
    raw_data_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\test'
    logs_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\logFiles'
    user_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\cleanedUsers'
    dataframe_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes'

    # TODO for all users
    # logs_test_user = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\logFiles\testxx'
    # import_json(logs_test_user, dataframe_dir)

    # extract user files without logs
    extract_sessions(fr'{dataframe_dir}\testxx')
