import pandas as pd
import math
import json
import pathlib
import pickle
import numpy as np


def extract_sessions(df_logs):
    """
    Extract all the usage session into an extra dataframe with timsetamps, id and length and label all logs within that session with an id
    :param df_logs: the log dataframe to extract the sessions from
    :return: a tuple of the modified logs df with the session ids and the sesssions df
    """
    print("__extract sessions__")
    # Get dataframe with all Screen and boot events
    # Try without values?
    df_logs['session_id'] = ""
    df_logs['session_count'] = ""
    study_id = df_logs['studyID'].iloc[0]
    last_ts = df_logs['correct_timestamp'].iloc[-1]
    last_id = df_logs['id'].iloc[-1]

    # Usage events screen interactive??
    screen_logs = df_logs[(df_logs['eventName'].values == 'SCREEN') | (df_logs['event'].values == 'SHUTDOWN')]
    # drop unnessesary events

    # screen_logs.drop(screen_logs.index[screen_logs['event'] == 'UNKNOWN'], inplace=True)
    # screen_logs.drop(screen_logs.index[screen_logs['event'] == 'ON_UNLOCKED'], inplace=True)

    count = 1
    sessions = []  # pd.DataFrame(columns=['session_id', 'count', 'session_length', 'timestamp_1', 'timestamp_2'])

    # get all timestamps 1: User ON_USERPRESENT  (SCREEN_ON_UNLOCKED?)
    timestamps_1 = screen_logs[screen_logs['event'].values == 'ON_USERPRESENT']['correct_timestamp']

    # len(df.index) == 0 - no rows
    # df.dropna().empty
    while not timestamps_1.empty:
        print(len(timestamps_1))
        timestamp_1 = timestamps_1.values[0]

        # TODO check if there is no ts2 event before the next ts1 event
        # find ts2: OFF_LOCKED, OFF_UNLOCKED, TODO everything else that is not ON_USerpresent? und timestamp davon is later than tsi
        timestamps_2 = screen_logs[((screen_logs['event'].values == 'OFF_LOCKED') |
                                    (screen_logs['event'].values == 'OFF_UNLOCKED') |
                                    (screen_logs['event'].values == 'SHUTDOWN')) &
                                   (screen_logs['correct_timestamp'] > timestamp_1)]

        # get last timestamp of log list if ts2 is empty
        timestamp_2 = timestamps_2['correct_timestamp'].values[0] if not timestamps_2.empty else last_ts

        ts_between = screen_logs[(screen_logs['event'].values == 'ON_USERPRESENT') &
                                 (screen_logs['correct_timestamp'] > timestamp_1) &
                                (screen_logs['correct_timestamp'] < timestamp_2)]

        if ts_between.empty:
            id_saved = timestamps_2['id'].values[0] if not timestamps_2.empty else last_id
            # print(f'ts2 {timestamp_2}')

            # calculate length of session in ms
            session_length = (timestamp_2 - timestamp_1)  # / np.timedelta64(1, 'ms')
            # print(f'sesionlength {session_length}')

            # assign sessionID: count and timestamp? #TODO only use count?
            session_id = f'{count}-{study_id}'
            # print(f'sessionId {session_id}')

            # assign sessionID to all these rows within session timestamps
            df_logs.loc[((df_logs['correct_timestamp'].values >= timestamp_1) & (
                    df_logs['correct_timestamp'].values <= timestamp_2)), 'session_id'] = count

            # also find the ESM Lock for that session and label it
            df_logs.loc[((df_logs['correct_timestamp'].values >= timestamp_1) &
                         (df_logs['id'].values == id_saved) &
                         (df_logs['eventName'].values == 'ESM')), 'session_id'] = count


            # add a new row to sessiondf with sessionid, sessionlngth, first and last timestamp
            sessions.append({'session_id': session_id, 'count': count, 'studyID': study_id, 'session_length': session_length, 'timestamp_1': timestamp_1,
                             'timestamp_2': timestamp_2})
            # increment count
            count += 1

            # get the new ts1 where timestamp is larger than ts2
            timestamps_1 = screen_logs[(screen_logs['event'].values == 'ON_USERPRESENT') & (
                           screen_logs['correct_timestamp'].values > timestamp_2)]['correct_timestamp']
        else:
            # get the new ts1 where timestamp is larger than ts2
            timestamps_1 = screen_logs[(screen_logs['event'].values == 'ON_USERPRESENT') & (
                           screen_logs['correct_timestamp'].values > timestamp_1)]['correct_timestamp']

        # print(f'new ts1 {timestamps_1.head()}')

    sessions_list = pd.DataFrame(sessions)
    print("....extract sessions finished....")
    return df_logs, sessions_list


# discard session that are very small/very big?

def extract_meta(df_logs):
    test = df_logs['metaData'].apply(pd.Series)
    print(test.columns.values)
    df_logs.drop(columns=['metaData'], inplace=True)
    t = pd.concat([df_logs, test], axis=1)
    print(t.columns.values)
    t.to_csv(fr'{dataframe_dir}\test.csv')


if __name__ == '__main__':
    dataframe_dir_logs = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\test'
    dataframe_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes'
    pathlist = pathlib.Path(dataframe_dir_logs).glob('**/*.pickle')

    user_sessions = {}
    for data_path in pathlist:
        path_in_str = str(data_path)
        print(path_in_str)
        df_logs = pd.read_pickle(path_in_str)

        # ['description' 'event' 'eventName' 'id' 'name' 'packageName' 'timestamp' 'timezoneOffset' 'metaData' 'studyID' 'correct_timestamp' 'weekday']

        extracted = extract_sessions(df_logs)  # can i assign to same variable again?
        user_sessions[data_path.stem] = extracted[1]

        with open(fr'{dataframe_dir_logs}\{data_path.stem}.pickle', 'wb') as f:
            pickle.dump(extracted[0], f, pickle.HIGHEST_PROTOCOL)

    # TODO save all in extra files or one file again?
    with open(fr'{dataframe_dir}\user-sessions.pickle', 'wb') as f:
        pickle.dump(user_sessions, f, pickle.HIGHEST_PROTOCOL)
