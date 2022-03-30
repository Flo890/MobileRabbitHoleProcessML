import pandas as pd
import math
import pathlib
import pickle
import numpy as np


def extract_sessions(df_logs, study_id):
    print("__extract sessions__")
    # Get dataframe with all Screen and boot events
    # Try without values?
    df_logs['session_id'] = ""

    # Usage events screen interactive??
    screen_logs = df_logs[(df_logs['eventName'].values == 'SCREEN') | (df_logs['event'].values == 'SHUTDOWN')]
    # drop unnessesary events
    # screen_logs = screen_logs.drop(screen_logs.index[screen_logs['event'] == 'UNKNOWN'])
    # screen_logs = screen_logs.drop(screen_logs.index[screen_logs['event'] == 'ON_UNLOCKED'])

    count = 1
    sessions = []  # pd.DataFrame(columns=['session_id', 'session_length', 'timestamp_1', 'timestamp_2'])

    # get all timestamps 1: User ON_USERPRESENT  (SCREEN_ON_UNLOCKED?)
    timestamps_1 = screen_logs[screen_logs['event'].values == 'ON_USERPRESENT']['correct_timestamp']

    # While all ts1 are not nan
    # len(df.index) == 0 - no rows
    # or df. empty -> no values
    # df.dropna().empty
    # print(type(timestamps_1))
    # print(np.isnan(timestamp_1.isna))

    while not timestamps_1.empty:
        # while not pd.isna(timestamps_1):
        # print(type(timestamps_1))
        # timestamp_1 = timestamps_1.values[0]
        # print(len(timestamps_1))
        timestamp_1 = timestamps_1.values[0]
        # print(f'ts1 {timestamp_1}')

        # find ts2: OFF_LOCKED, OFF_UNLOCKED, TODO everything else that is not ON_USerpresent? und timestamp davon is later than tsi
        # if tss is nan, take max timestamp of that data
        timestamps_2 = screen_logs[((screen_logs['event'].values == 'OFF_LOCKED') |
                                    (screen_logs['event'].values == 'OFF_UNLOCKED') |
                                    (screen_logs['event'].values == 'SHUTDOWN')) &
                                   (screen_logs['correct_timestamp'] > timestamp_1)]['correct_timestamp']
        # get last timestamp of log list if ts2 is empty

        timestamp_2 = timestamps_2.values[0] if not timestamps_2.empty else df_logs['correct_timestamp'][-1]
        # print(f'ts2 {timestamp_2}')

        # calculate length of session in ms
        session_length = (timestamp_2 - timestamp_1) / np.timedelta64(1, 'ms')
        # print(f'sesionlength {session_length}')

        # assign sessionID: count and timestamp? #TODO only use count?
        session_id = f'{count}-{study_id}'
        # print(f'sessionId {session_id}')

        # get all data where timestamp is smaller thatn ts1 but larger than ts 1
        # assign sessionID to all these rows with sessionid

        df_logs.loc[((df_logs['correct_timestamp'].values >= timestamp_1) & (
                    df_logs['correct_timestamp'].values <= timestamp_2)), 'session_id'] = session_id

        # add a new row to sessiondf with sessionid, sessionlngth, first and last timestamp
        sessions.append({'session_id': timestamp_2, 'session_length': session_length, 'timestamp_1': timestamp_1,
                         'timestamp_2': timestamp_2})

        # increment count
        count += 1
        # assign new timestamp
        # nonlocal timestamps_1
        timestamps_1 = screen_logs[(screen_logs['event'].values == 'ON_USERPRESENT') & (screen_logs['correct_timestamp'].values > timestamp_2)]['correct_timestamp']
        # print(f'new ts1 {timestamps_1.head()}')


        # get the new ts1 where timestamp is larger than ts2

    sessions_list = pd.DataFrame(sessions)
    print("finished")
    return df_logs, sessions_list


# OR:
# Group with session id, loop over groups and do steps above (with exption that USER_present is dumbly in session previously
def extract_sessions_alt(path_to_pickel):
    df_logs = pd.read_pickle(path_to_pickel)
    df_logs.drop(df_logs.index[df_logs['id'] == ''], inplace=True)
    df_logs.drop(df_logs.index[df_logs['event'] == 'TYPE_WINDOW_CONTENT_CHANGED'], inplace=True)
    session_grouped = df_logs.groupby(['id'])
    # session_grouped_cut = session_grouped.apply(cut_sessions)


# discard session that are very small?


if __name__ == '__main__':
    dataframe_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\test'
    pathlist = pathlib.Path(dataframe_dir).glob('**/*.pickle')

    for data_path in pathlist:
        path_in_str = str(data_path)
        df_logs = pd.read_pickle(path_in_str)
        # print("______________________")
        # print(data_path.stem)

        df_logs_t = extract_sessions(df_logs, study_id=data_path.stem)  # can i assign to same variable again?
        print(df_logs_t[0]['session_id'].head())
        # print(df_logs_t.head())
        # print(df_logs_t.columns.values)
