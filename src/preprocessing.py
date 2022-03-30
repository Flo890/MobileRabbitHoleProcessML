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
raw_data_user = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\live\2022-03-25T00 44 32Z_1Uhr46absentmindedtrack-default-rtdb_data.json.gz'
logs_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\logFiles'
user_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\cleanedUsers'
dataframe_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\test'


def extractData():
    # only run once
    # getFromJson.extractUsers(file_path=raw_data_user, end_directory=user_dir, is_gzip=True)
    getFromJson.extract_logs(directory=raw_data_dir, end_directory=dataframe_dir, save_type=3, is_gzip=True)


def preprocessing():
    pathlist = pathlib.Path(logs_dir).glob('**/*.pickle')

    for data_path in pathlist:
        path_in_str = str(data_path)
        df_logs = pd.read_pickle(path_in_str)

        df_logs_t = prepareTimestamps.process_timestamps(df_logs) # can i assign to same variable again?
       #  df_logs_sessions = extractSessions.extract_sessions(df_logs_t)


if __name__ == '__main__':

    extractData()
    preprocessing()
