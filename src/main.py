import datetime
import json
import pathlib
import re
import pandas as pd
from cleanData import CleanData


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

        df['timestamp'] = pd.to_datetime(df['timestamp'], yearfirst=True, unit='ms')
        df.reset_index().set_index(df['timestamp'], drop=False)
        # or df.reset_index().set_index(df['timestamp'], drop=False) to keep old index as colum
        # TODO timezoneoffset

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
    # temp_session = logs.loc[logs['id']]
    # temp_session = logs.groupby("id")
    temp_session = logs[logs.duplicated(['id'])]
    print(temp_session[['id']].head(10))
    print(temp_session.head(5))
    print(temp_session.size)
    print(logs.size)


if __name__ == '__main__':
    raw_data_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\test'
    logs_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\logFiles'
    user_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\cleanedUsers'
    dataframe_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes'

    # extract all logs for each user
    # cleanData = CleanData()
    # cleanData.extractLogs(directory=raw_data_dir, end_directory=logs_dir)

    # TODO for all users
    #logs_test_user = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\logFiles\testxx'
    # import_json(logs_test_user, dataframe_dir)


    # extract user files without logs
    extract_sessions(fr'{dataframe_dir}\testxx')
