import datetime
import json
import pathlib
import re
import pandas as pd
from cleanData import CleanData


def import_json(directory):
    """
    Import the json logs form one user into a dataframe and saves it as pickel.
    :param directory: the directory where the json log files are located
    """
    print("import json logs")
    pathlist = pathlib.Path(directory).glob('**/*.json')
    for data_path in pathlist:
        path_in_str = str(data_path)

        # with open(path_in_str, encoding='utf-8') as data:
        #     d = json.load(data)

        # read the json to dataFrame
        logs = pd.read_json(path_in_str, orient="index")

        # TODO maybe delete all duplicate rows?
        # TODO extract metaData?
        # logsmeta = extractMetaData(logs)


if __name__ == '__main__':
    raw_data_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\test'
    logs_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\logFiles'
    user_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\cleanedUsers'

    # extract all logs for each user
    cleanData = CleanData()
    cleanData.extractLogs(directory=raw_data_dir, end_directory=logs_dir)

    # extract user files without logs
    # extractUsers(user_dir)
