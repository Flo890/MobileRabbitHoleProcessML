import pandas as pd
import numpy as np
import tldextract
import ML_helpers
import grouped_sessions
import json





def get_features_for_sessiongroup(df_logs, df_sessions, df_session_groups):
    """
    like :get_features_for_session, but calculating features for a group of sessions
    :param df_logs: the logs dataframe of one user
    :param df_sessions: the extracted session dataframe of one user
    :param df_session_groups: result from grouped_sessions.session_sessions_to_aggregate_df
    :return: the sessions and feature dataframe for one user
    """
    # Add demographics features - equivalent
    path_questionnaire_1 = r'C:\projects\rabbithole\RabbitHoleProcess\data\rawData'
    df_MRH1 = pd.read_csv(f'{path_questionnaire_1}\MRH1.csv', sep=',')
    studyID = df_logs['studyID'].values[0]
    if not df_MRH1[df_MRH1['IM01_01'].values == studyID].empty:
        df_qu_user = df_MRH1[df_MRH1['IM01_01'].values == studyID].index.item()
        df_session_groups['f_demographics_gender'] = df_MRH1.loc[df_qu_user, 'SD01']  # where 1: female, 2: male, 3: non-binary, 4: Prefer not to disclose, 5: Other
        df_session_groups['f_demographics_age'] = df_MRH1.loc[df_qu_user, 'SD02_01']

        df_session_groups['f_absentminded_use'] = df_MRH1.loc[df_qu_user, df_MRH1.columns.str.startswith('AB01')].mean()
        df_session_groups['f_general_use'] = df_MRH1.loc[df_qu_user, df_MRH1.columns.str.startswith('AB02')].mean()
        # WLAN name IM06_01
    else:
        df_session_groups['f_demographics_gender'] = 0  # where 1: female, 2: male, 3: non-binary, 4: Prefer not to disclose, 5: Other
        df_session_groups['f_demographics_age'] = 0
        df_session_groups['f_absentminded_use'] = 0
        df_session_groups['f_general_use'] = 0

    # Prepare feature columns
    pd.to_datetime(df_session_groups['timestamp_1'])  # will hold group start / group end
    pd.to_datetime(df_session_groups['timestamp_2'])

    # TODO aggregate ESM answers
    # TODO replace all the following df_sessions with df_session_groups
    df_sessions['f_esm_intention'] = ""
    df_sessions['f_esm_finished_intention'] = ""
    df_session_groups['f_esm_more_than_intention'] = "[]"
    df_session_groups['f_esm_atleastone_more_than_intention'] = ""
    df_sessions['f_esm_track_of_time'] = ""
    df_sessions['f_esm_track_of_space'] = ""
    df_sessions['f_esm_emotion'] = ""
    df_session_groups['f_esm_regret'] = "[]"
    df_session_groups['f_esm_atleastone_regret'] = ""
    df_sessions['f_esm_agency'] = ""

    df_sessions['f_sequences'] = np.nan   # can be concatenated
    df_session_groups['f_hour_of_day'] = np.nan
    df_session_groups['f_weekday'] = np.nan
    df_session_groups['f_session_group_timespan'] = np.nan  # whole time that passed from beginning till end, including breaks with phone locked
    df_session_groups['f_session_group_length_active'] = np.nan  # amount of time that the user was active on the phone
    df_session_groups['f_session_group_length_active_mean'] = np.nan
    df_session_groups['f_session_group_length_active_median'] = np.nan
    df_session_groups['f_session_group_length_active_sd'] = np.nan
    df_session_groups['f_time_since_last_session'] = 0
    df_session_groups['f_glances_since_last_session'] = 0
    df_session_groups['f_count_session_1h'] = 0
    df_session_groups['f_count_session_2h'] = 0
    df_session_groups['f_count_session_3h'] = 0
    df_sessions['f_clicks'] = 0
    df_sessions['f_scrolls'] = 0
    df_sessions['f_internet_connected_WIFI'] = 0
    df_sessions['f_internet_connected_mobile'] = 0
    df_sessions['f_internet_disconnected'] = 0
    df_sessions['f_internet_else'] = 0
    df_sessions['f_ringer_mode_silent'] = 0
    df_sessions['f_ringer_mode_vibrate'] = 0
    df_sessions['f_ringer_mode_normal'] = 0
    df_sessions['f_ringer_mode_unknown'] = 0

    currentInternetState = {'connectionType': 'UNKNOWN_first', 'wifiState': 'UNKNOWN_first', 'wifiName': '', 'timestamp': np.nan}
    currentApp = {"packageName": 'UNKNOWN_first', "timestamp": np.nan}
    current_sequence_list = []
    current_app_sequence_list = []
    currentRingerMode = {'mode': 'UNKNOWN_first', 'timestamp': np.nan}
    currentActivity = {'activity': 'UNKNOWN_first', 'transition': 'UNKNOWN_first', 'timestamp': np.nan}

    bag_of_apps_vocab = []

    # grouped_logs = df_logs.groupby('session_id').agg({'count': sum})
    grouped_logs = df_logs.groupby(['studyID', 'group_id'])   # here we group by group, not by session.
    # TODO the following loop can then mostly stay the same

    # Iterate over groups (instead of sessions)
    group_counter = 0
    for name, df_group in grouped_logs:
        if group_counter > 20: break
        group_counter += 1
        if name[1]:  # | (not pd.isna(name):# if name is not empty
            # Get the df_session row of the current session to assign the values to
            df_row = df_session_groups[(df_session_groups['group_id'].values == name[1])]
            index_row = df_session_groups[(df_session_groups['group_id'].values == name[1])].index.item()
            current_session_start = df_row['timestamp_1'].iloc[0]

            # ---------------  FEATURE TIME AND GLANCES SINCE LAST SESSION ------------------ #
            index_last_session = index_row - 1
            if index_last_session > 0:
                last_session_end = df_session_groups['timestamp_2'].iloc[index_last_session]
                df_session_groups.loc[index_row, 'f_time_since_last_session'] = round(
                    (current_session_start - last_session_end).total_seconds() * 1000)

                # GLances: get all screen ON_LOCKED events between last session ts and current session ts
                glances_count = len(df_logs[(df_logs['event'].values == 'ON_LOCKED') &
                                            (df_logs['correct_timestamp'].values > last_session_end) &
                                            (df_logs['correct_timestamp'].values < current_session_start)])
                df_session_groups.loc[index_row, 'f_glances_since_last_session'] = df_session_groups.loc[
                                                                                 index_row, 'f_glances_since_last_session'] + glances_count

            # ---------------  FEATURE COUNT SESSION IN LAST HOUR ------------------ #
            # Calculate timestamp one hour ago
            ts_one_hour = current_session_start - pd.Timedelta(hours=1)
            # Get session that started within the last hour
            df_session_groups.loc[index_row, 'f_count_session_1h'] = len(
                    df_session_groups[(df_session_groups['timestamp_1'].values > ts_one_hour) &
                                (df_session_groups['timestamp_2'].values < current_session_start)])

            ts_two_hour = current_session_start - pd.Timedelta(hours=2)
            # Get session that started within the last hour
            df_session_groups.loc[index_row, 'f_count_session_2h'] = len(
                    df_session_groups[(df_session_groups['timestamp_1'].values > ts_two_hour) &
                                (df_session_groups['timestamp_2'].values < current_session_start)])

            ts_three_hour = current_session_start - pd.Timedelta(hours=3)
            # Get session that started within the last hour
            df_session_groups.loc[index_row, 'f_count_session_3h'] = len(
                    df_session_groups[(df_session_groups['timestamp_1'].values > ts_three_hour) &
                                (df_session_groups['timestamp_2'].values < current_session_start)])

            # ---------------  FEATURE HOUR OF DAY ------------------ #
            # df_row['f_hour_of_day'] = df_row['timestamp_1'].hour
            df_session_groups.loc[index_row, 'f_hour_of_day'] = df_row['timestamp_1'].iloc[0].hour

            # ---------------  FEATURE WEEKDAY ------------------ #
            # df_row['f_weekday'] = df_row['timestamp_1'].dt.dayofweek
            df_session_groups.loc[index_row, 'f_weekday'] = df_row['timestamp_1'].iloc[0].dayofweek

            # ---------------  FEATURE SESSION GROUP LENGTH------------------ #
            # df_row['f_session_length'] = df_row['timestamp_2'] - df_row['timestamp_1']
            df_session_groups.loc[index_row, 'f_session_group_timespan'] = df_row['timestamp_2'].iloc[0] - \
                                                                 df_row['timestamp_1'].iloc[0]

            # ---------------  FEATURE SESSION LENGTH------------------ #
            #  aggregate metrics over sessions
            session_durations_active = []
            # get sessions in this group
            for session_id_i in range(len(df_session_groups.loc[index_row, 'session_ids'])):
                session_id = df_session_groups['session_ids'].iloc[index_row][session_id_i].iloc[0]
                a_session = df_sessions.loc[df_sessions['session_id'] == session_id]
                a_session_length = a_session['timestamp_2'].iloc[0] - a_session['timestamp_1'].iloc[0]
                session_durations_active.append(a_session_length)
            df_session_groups.loc[index_row, 'f_session_group_length_active'] = pd.Series(session_durations_active).sum()
            df_session_groups.loc[index_row, 'f_session_group_length_active_mean'] = pd.Series(session_durations_active).mean()
            df_session_groups.loc[index_row, 'f_session_group_length_active_median'] = pd.Series(session_durations_active).median()
            df_session_groups.loc[index_row, 'f_session_group_length_active_sd'] = pd.Series(session_durations_active).std()

        # --------------- FEATURE ESM ------------------ #
        for log in df_group.itertuples():

            # --------------- FEATURE ESM ------------------ #
            if log.eventName == 'ESM':
                if log.event == 'ESM_LOCK_Q_MORE':
                    print("more than intention: " + str(index_row) + ": " + log.name)
                    my_json = json.loads(df_session_groups.loc[index_row, 'f_esm_more_than_intention'])
                    my_json.append(log.name)
                    df_session_groups.loc[index_row, 'f_esm_more_than_intention'] = json.dumps(my_json)

                elif log.event == 'ESM_LOCK_Q_REGRET':
                    print("regret: " + str(index_row) + ": " + log.name)
                    my_json = json.loads(df_session_groups.loc[index_row, 'f_esm_regret'])
                    my_json.append(log.name)
                    df_session_groups.loc[index_row, 'f_esm_regret'] = json.dumps(my_json)

        ## atleast-one aggregation more than intention
        my_json = json.loads(df_session_groups.loc[index_row, 'f_esm_more_than_intention'])
        group_mti = 'No'
        for esm_a in my_json:
            if esm_a == 'Yes':
                group_mti = 'Yes'

        if len(my_json) > 0:
            df_session_groups.loc[index_row, 'f_esm_atleastone_more_than_intention'] = group_mti

        ## atleast-one aggregation regret: group counts as regret, if at least one session has ESM label regret
        my_json = json.loads(df_session_groups.loc[index_row, 'f_esm_regret'])
        group_regret = 'No'
        for esm_a in my_json:
            if float(esm_a) >= 4.0:  # if regret of 4.0 or higher occurs, count group as regretted
                group_regret = 'Yes'

        if len(my_json) > 0: # if no ESM was answers, keep NA
            df_session_groups.loc[index_row, 'f_esm_atleastone_regret'] = group_regret

    df_session_groups_sorted = df_session_groups.sort_values(by=['group_id'])  # to "remove gaps", when processing only n groups
    print("finished extracting features")
    return df_session_groups, bag_of_apps_vocab






if __name__ == '__main__':
    df = pd.read_pickle(
        "C:\\projects\\rabbithole\\RabbitHoleProcess\\data\\dataframes\\sessions_with_features\\AN23GE.pickle")
    res = grouped_sessions.build_session_sessions(df, 120)

    res2 = grouped_sessions.session_sessions_to_aggregate_df(res)

    df_logs = pd.read_pickle("C:\\Users\\florianb\\Downloads\\AN23GE.pickle")
      #  "D:\\usersorted_logs_preprocessed\\AN23GE.pickle")  # AN23GE.pickle")   AN09BI    LE13FO

    ### add column group_id to df_logs
    # create mapping session_id -> group_id
    mapping = dict()
    counter = 0
    for a_group_i in range(len(res2)):
        a_group = res2.iloc[a_group_i]
        for a_session in a_group['session_ids']:
            mapping[a_session[counter].split("-")[0]] = a_group['group_id']
            counter += 1
    mapping[''] = None

    df_logs['group_id'] = df_logs['session_id'].apply(lambda x: mapping[str(x)])

    df_all_sessions = pd.read_pickle(
        "C:\\projects\\rabbithole\\RabbitHoleProcess\\data\\dataframes\\sessions_raw_extracted\\AN23GE-sessions.pickle")

    df_session_group_features = get_features_for_sessiongroup(df_logs, df_all_sessions, res2)