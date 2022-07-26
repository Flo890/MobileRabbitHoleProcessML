import pandas as pd
import numpy as np
import tldextract
import ML_helpers
import grouped_sessions
import json
import warnings


def progress_bar(current, total, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)


def get_features_for_sessiongroup(df_logs, df_sessions, df_session_groups):
    """
    like :get_features_for_session, but calculating features for a group of sessions
    :param df_logs: the logs dataframe of one user
    :param df_sessions: the extracted session dataframe of one user
    :param df_session_groups: result from grouped_sessions.session_sessions_to_aggregate_df
    :return: the sessions and feature dataframe for one user
    """
    print('Starting feature extraction for session-group')
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

    df_session_groups['f_session_count_in_group'] = np.nan
    df_session_groups['f_sequences'] = np.nan   # can be concatenated
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
    df_session_groups['f_clicks'] = 0
    df_session_groups['f_scrolls'] = 0
    df_session_groups['f_internet_connected_WIFI'] = 0
    df_session_groups['f_internet_connected_mobile'] = 0
    df_session_groups['f_internet_disconnected'] = 0
    df_session_groups['f_internet_else'] = 0
    df_session_groups['f_ringer_mode_silent'] = 0
    df_session_groups['f_ringer_mode_vibrate'] = 0
    df_session_groups['f_ringer_mode_normal'] = 0
    df_session_groups['f_ringer_mode_unknown'] = 0

    df_session_groups['f_activity_resumed_count'] = 0
    df_session_groups['f_activity_resumed_frequency'] = 0
    df_session_groups['f_click_frequency'] = 0.0
    df_session_groups['f_scroll_frequency'] = 0.0

    currentInternetState = {'connectionType': 'UNKNOWN_first', 'wifiState': 'UNKNOWN_first', 'wifiName': '', 'timestamp': np.nan}
    currentApp = {"packageName": 'UNKNOWN_first', "timestamp": np.nan}
    current_sequence_list = []
    current_app_sequence_list = []
    currentRingerMode = {'mode': 'UNKNOWN_first', 'timestamp': np.nan}
    currentActivity = {'activity': 'UNKNOWN_first', 'transition': 'UNKNOWN_first', 'timestamp': np.nan}

    bag_of_apps_vocab = []

    # grouped_logs = df_logs.groupby('session_id').agg({'count': sum})
    grouped_logs = df_logs.groupby(['studyID', 'group_id'])   # here we group by group, not by session.

    # Iterate over groups (instead of sessions)
    group_counter = 0
    for name, df_group in grouped_logs:
        if group_counter > 20: break  # TODO limits amount of groups for development here
        group_counter += 1
        progress_bar(group_counter,len(grouped_logs))
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

            # ---------------  FEATURE SESSION COUNT ---------------
            df_session_groups.loc[index_row, 'f_session_count_in_group'] = len(df_session_groups.loc[index_row, 'session_ids'])

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
                 #   print("more than intention: " + str(index_row) + ": " + log.name)
                    my_json = json.loads(df_session_groups.loc[index_row, 'f_esm_more_than_intention'])
                    my_json.append(log.name)
                    df_session_groups.loc[index_row, 'f_esm_more_than_intention'] = json.dumps(my_json)

                elif log.event == 'ESM_LOCK_Q_REGRET':
                  #  print("regret: " + str(index_row) + ": " + log.name)
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


        # loop over all logs from a group
        for log in df_group.sort_values(by=['correct_timestamp']).itertuples():
            # ---------------  FEATURE SCROLL AND CLICK COUNT ------------------ #
            if (log.eventName == 'ACCESSIBILITY') | (log.event == 'ACCESSIBILITY_KEYBOARD_INPUT'):

                if log.event == 'TYPE_VIEW_CLICKED':
                    df_session_groups.loc[index_row, 'f_clicks'] += 1
                    if f'f_clicks_{log.packageName}' not in df_session_groups.columns:
                        df_session_groups[f'f_clicks_{log.packageName}'] = 0

                    pre = df_session_groups.loc[index_row, f'f_clicks_{log.packageName}'] + 1
                    df_session_groups.loc[index_row, f'f_clicks_{log.packageName}'] = pre

                    # ----- save clicks per app category ------#
                    app_category = ML_helpers.get_app_category(log.packageName)
                    if f'f_clicks_app_category_{app_category}' not in df_session_groups.columns:
                        df_session_groups[f'f_clicks_app_category_{app_category}'] = 0

                    pre_count_cat = df_session_groups.loc[index_row, f'f_clicks_app_category_{app_category}'] + 1
                    df_session_groups.loc[index_row, f'f_clicks_app_category_{app_category}'] = pre_count_cat

                elif log.event == 'TYPE_VIEW_SCROLLED':
                    df_session_groups.loc[index_row, 'f_scrolls'] += 1
                    if f'f_scrolls_{log.packageName}' not in df_session_groups.columns:
                        df_session_groups[f'f_scrolls_{log.packageName}'] = 0

                    pre = df_session_groups.loc[index_row, f'f_scrolls_{log.packageName}'] + 1
                    df_session_groups.loc[index_row, f'f_scrolls_{log.packageName}'] = pre

                    # ----- save scrolls per app category ------ #
                    app_category = ML_helpers.get_app_category(log.packageName)
                    if f'f_scrolls_app_category_{app_category}' not in df_session_groups.columns:
                        df_session_groups[f'f_scrolls_app_category_{app_category}'] = 0

                    pre_count_cat = df_session_groups.loc[index_row, f'f_scrolls_app_category_{app_category}'] + 1
                    df_session_groups.loc[index_row, f'f_scrolls_app_category_{app_category}'] = pre_count_cat

            # ---------------  FEATURE SEQUENCE BrowserURL------------------ #
            elif log.eventName == 'ACCESSIBILITY_BROWSER_URL':
                # eventName.ACCESSIBILITY_BROWSER_URL,
                # event = getEventType(event) (accesibility),
                # description = browserUrl,
                # name = "BrowserURL",
                # packageName = packageName
                seq = ('BROWSER_URL', get_domain_from_url(log.description))
                current_sequence_list.append(seq)

                if 'f_browser_url' not in df_session_groups.columns:
                    df_session_groups['f_browser_url'] = 0

                pre = df_session_groups.loc[index_row, 'f_browser_url'] + 1
                df_session_groups.loc[index_row, 'f_browser_url'] = pre

            # ---------------  FEATURE NOTIFICATION ------------------ #
            elif log.eventName == 'NOTIFICATION':
                #  MetaNotification(
                #  priority, category, infoText, subText, interaction = action.name)
                # LogEvent(LogEventName.NOTIFICATION, timestamp, event = title, description = text, packageName = packageName)
                current_sequence_list.append((log.interaction, log.packageName))

            # ---------------  FEATURE INTERNET CONNECTION ------------------ #
            elif log.eventName == 'INTERNET':
                new_state = {'connectionType': log.description,
                             'wifiState': log.event,
                             'wifiName': log.name,
                             'timestamp': log.correct_timestamp}

                if (currentInternetState['connectionType'] != new_state['connectionType']) | (
                        (currentInternetState['connectionType'] == new_state['connectionType']) & (
                        currentInternetState['wifiName'] != new_state['wifiName'])):
                    # Calculate time difference between current and new state
                    if currentInternetState['connectionType'] == 'UNKNOWN_first':
                        currentInternetState = new_state

                    else:
                        dif = new_state['timestamp'] - currentInternetState['timestamp']
                        dif = round(dif.total_seconds() * 1000)
                        # add old state to feature table
                        if currentInternetState['connectionType'] == 'CONNECTED_WIFI':
                            df_session_groups.loc[index_row, 'f_internet_connected_WIFI'] += dif

                            #  Also add the tme spent in specific wifi network
                            wifi = currentInternetState['wifiName']
                            if f'f_internet_connected_WIFI_{wifi}' not in df_session_groups.columns:
                                df_session_groups[f'f_internet_connected_WIFI_{wifi}'] = 0  # pd.Timedelta(days=0)

                            pre = df_session_groups.loc[index_row, f'f_internet_connected_WIFI_{wifi}'] + dif
                            df_session_groups.loc[index_row, f'f_internet_connected_WIFI_{wifi}'] = pre

                        elif currentInternetState['connectionType'] == 'CONNECTED_MOBILE':
                            df_session_groups.loc[index_row, 'f_internet_connected_mobile'] += dif

                        elif currentInternetState['connectionType'] == 'UNKNOWN':
                            df_session_groups.loc[index_row, 'f_internet_disconnected'] += dif

                        else:
                            df_session_groups.loc[index_row, 'f_internet_else'] += dif
                        # elif currentInternetState['connectionType'] == 'CONNECTED_ETHERNET':
                        # elif currentInternetState['connectionType'] == 'CONNECTED_VPN':

                        # Save the new state as current state
                        currentInternetState = new_state

            # ---------------  FEATURE RINGER MODE ------------------ #
            elif log.eventName == 'RINGER_MODE':
                new_mode = {'mode': log.event, 'timestamp': log.correct_timestamp}

                if currentRingerMode['mode'] != new_mode['mode']:
                    # Calculate time difference between current and new state
                    if currentRingerMode['mode'] == 'UNKNOWN_first':
                        currentRingerMode = new_mode

                    else:
                        # Check if cached ringer mode,i if yes take session start if not take chached timestamp
                        # print(new_mode['timestamp'], currentRingerMode['timestamp'], current_session_start)
                        if current_session_start > currentRingerMode['timestamp']:
                            dif = round((new_mode['timestamp'] - current_session_start).total_seconds() * 1000)
                        else:
                            dif = round(
                                (new_mode['timestamp'] - currentRingerMode['timestamp']).total_seconds() * 1000)

                        # add old state to feature table
                        if currentRingerMode['mode'] == 'NORMAL_MODE':
                            df_session_groups.loc[index_row, 'f_ringer_mode_normal'] += dif

                        elif currentRingerMode['mode'] == 'VIBRATE_MODE':
                            df_session_groups.loc[index_row, 'f_ringer_mode_vibrate'] += dif

                        elif currentRingerMode['mode'] == 'SILENT_MODE':
                            df_session_groups.loc[index_row, 'f_ringer_mode_silent'] += dif

                        elif currentRingerMode['mode'] == 'UNKNOWN':
                            df_session_groups.loc[index_row, 'f_ringer_mode_unknown'] += dif

                        # Save the new state as current state
                        currentRingerMode = new_mode

            # ---------------  FEATURE SEQUENCE PHONE ------------------ #
            elif log.eventName == 'PHONE':
                # eventName = LogEventName.PHONE,
                # timestamp = timestamp,
                # event = event,
                # description = duration.toString()
                current_sequence_list.append((log.eventName, log.event))

            # ---------------  FEATURE SEQUENCE SMS ------------------ #
            elif log.eventName == 'SMS':
                # MetaSMS(
                # phoneNumber = numberHashed,
                # countryCode = countryCode,
                # length = messageLength,
                # messageHash = messageHash,
                # event = type.name,
                # description = smsID,
                current_sequence_list.append((log.eventName, log.event))

            # ---------------  FEATURE USED APPS + SEQUENCE ------------------ #
            elif log.eventName == 'USAGE_EVENTS':
                # check the package name if different from before
                # Ignore other events that activity resumed
                # Check if it is activity resumed (ignore stopped and paused) when package name different and take this as endpoint for previous
                if (currentApp['packageName'] != log.packageName) & (log.event == 'ACTIVITY_RESUMED'):

                    df_session_groups['f_activity_resumed_count'] += 1

                    # Save the new app to sequence lists
                    current_sequence_list.append(('APP', log.packageName))
                    current_app_sequence_list.append(log.packageName)

                    if log.packageName not in bag_of_apps_vocab:
                        bag_of_apps_vocab.append(log.packageName)

                    if currentApp['packageName'] == 'UNKNOWN_first':
                        currentApp = {"packageName": log.packageName, "timestamp": log.correct_timestamp}
                    else:
                        # Save the new app to sequence list
                        # current_sequence_list.append(('APP', log.packageName))
                        # ___________________________________________________________________________________________#
                        # Save current app count
                        packagename = currentApp['packageName']

                        if f'f_app_count_{packagename}' not in df_session_groups.columns:
                            df_session_groups[f'f_app_count_{packagename}'] = 0

                        pre_count = df_session_groups.loc[index_row, f'f_app_count_{packagename}'] + 1
                        df_session_groups.loc[index_row, f'f_app_count_{packagename}'] = pre_count

                        # ___________________________________________________________________________________________#
                        # Save time spent on app
                        time_spent = round((log.correct_timestamp - currentApp['timestamp']).total_seconds() * 1000)

                        if f'f_app_time_{packagename}' not in df_session_groups.columns:
                            df_session_groups[f'f_app_time_{packagename}'] = 0  # pd.Timedelta(days=0)

                        pre_time = df_session_groups.loc[index_row, f'f_app_time_{packagename}'] + time_spent
                        df_session_groups.loc[index_row, f'f_app_time_{packagename}'] = pre_time

                        # ___________________________________________________________________________________________#
                        # save category count
                        app_category = ML_helpers.get_app_category(packagename)

                        if f'f_app_category_count_{app_category}' not in df_session_groups.columns:
                            df_session_groups[f'f_app_category_count_{app_category}'] = 0

                        pre_count_cat = df_session_groups.loc[index_row, f'f_app_category_count_{app_category}'] + 1
                        df_session_groups.loc[index_row, f'f_app_category_count_{app_category}'] = pre_count_cat

                        # ___________________________________________________________________________________________#
                        # save time spent on category
                        if f'f_app_category_time_{app_category}' not in df_session_groups.columns:
                            df_session_groups[f'f_app_category_time_{app_category}'] = 0  # pd.Timedelta(days=0)

                        pre_time_cat = df_session_groups.loc[
                                           index_row, f'f_app_category_time_{app_category}'] + time_spent
                        df_session_groups.loc[index_row, f'f_app_category_time_{app_category}'] = pre_time_cat

                        # ___________________________________________________________________________________________#
                        # save new app as current
                        currentApp = {"packageName": log.packageName, "timestamp": log.correct_timestamp}


        #### loop end ####
        # ---------------  SOME FREQUENCIES ---------------
        click_frequency = df_session_groups.loc[index_row, 'f_clicks'] / df_session_groups.loc[index_row, 'f_session_group_length_active'].seconds
        df_session_groups.loc[index_row, 'f_click_frequency'] = click_frequency

        scroll_frequency = df_session_groups.loc[index_row, 'f_scrolls'] / df_session_groups.loc[index_row, 'f_session_group_length_active'].seconds
        df_session_groups.loc[index_row, 'f_scroll_frequency'] = scroll_frequency

        activity_resumed_frequency = df_session_groups.loc[index_row, 'f_scrolls'] / df_session_groups.loc[index_row, 'f_session_group_length_active'].seconds
        df_session_groups.loc[index_row, 'f_activity_resumed_frequency'] = activity_resumed_frequency


        # ---------------  save the last internet and app state with the last log ------------------ #
        last_log_time = df_group['correct_timestamp'].iloc[-1]
        last_log_event = df_group['eventName'].iloc[-1]

        # ____________________________ save last internet sate ___________________________________#
        if (currentInternetState['connectionType'] != 'UNKNOWN_first') & (last_log_event != 'INTERNET'):
            dif = round((last_log_time - currentInternetState['timestamp']).total_seconds() * 1000)

            # add old state to feature table
            if currentInternetState['connectionType'] == 'CONNECTED_WIFI':
                df_session_groups.loc[index_row, 'f_internet_connected_WIFI'] += dif

                #  Also add the tme spent in specific wifi network
                wifi = currentInternetState['wifiName']
                if f'f_internet_connected_WIFI_{wifi}' not in df_session_groups.columns:
                    df_session_groups[f'f_internet_connected_WIFI_{wifi}'] = 0  # pd.Timedelta(days=0)

                pre = df_session_groups.loc[index_row, f'f_internet_connected_WIFI_{wifi}'] + dif
                df_session_groups.loc[index_row, f'f_internet_connected_WIFI_{wifi}'] = pre

            elif currentInternetState['connectionType'] == 'CONNECTED_MOBILE':
                df_session_groups.loc[index_row, 'f_internet_connected_mobile'] += dif

            elif currentInternetState['connectionType'] == 'UNKNOWN':
                df_session_groups.loc[index_row, 'f_internet_disconnected'] += dif
            currentInternetState = {'connectionType': 'UNKNOWN_first', 'wifiState': 'UNKNOWN_first', 'wifiName': '',
                                        'timestamp': np.nan}

        # ____________________________ save last ringer mode state ___________________________________#
        if (currentRingerMode['mode'] != 'UNKNOWN_first') & (last_log_event != 'RINGER_MODE'):
            #  print(last_log_time, currentRingerMode['timestamp'], current_session_start,  current_session_start > currentRingerMode['timestamp'])
            # Check if cached ringer mode,i if yes take session start if not take chached timestamp
            if current_session_start > currentRingerMode['timestamp']:
                dif = round((last_log_time - current_session_start).total_seconds() * 1000)
            else:
                dif = round((last_log_time - currentRingerMode['timestamp']).total_seconds() * 1000)
            # add old state to feature table
            if dif < 0:
                warnings.warn(f'ringermode dif below zero! {dif}')
            if currentRingerMode['mode'] == 'NORMAL_MODE':
                df_session_groups.loc[index_row, 'f_ringer_mode_normal'] += dif

            elif currentRingerMode['mode'] == 'VIBRATE_MODE':
                df_session_groups.loc[index_row, 'f_ringer_mode_vibrate'] += dif

            elif currentRingerMode['mode'] == 'SILENT_MODE':
                df_session_groups.loc[index_row, 'f_ringer_mode_silent'] += dif

            elif currentRingerMode['mode'] == 'UNKNOWN':
                df_session_groups.loc[index_row, 'f_ringer_mode_unknown'] += dif

            # Do not reset the cached ringer mode, as it only tiggered in change

        # _____________________________ save last app state _____________________________________#
        if (currentApp['packageName'] != 'UNKNOWN_first') & (last_log_event != 'USAGE_EVENTS'):
            packagename = currentApp['packageName']

            if f'f_app_count_{packagename}' not in df_session_groups.columns:
                df_session_groups[f'f_app_count_{packagename}'] = 0

            pre_count = df_session_groups.loc[index_row, f'f_app_count_{packagename}'] + 1
            df_session_groups.loc[index_row, f'f_app_count_{packagename}'] = pre_count

            # ___________________________________________________________________________________________#
            # Save time spent on app
            time_spent = round((last_log_time - currentApp['timestamp']).total_seconds() * 1000)

            if f'f_app_time_{packagename}' not in df_session_groups.columns:
                df_session_groups[f'f_app_time_{packagename}'] = 0  # pd.Timedelta(days=0)

            pre_time = df_session_groups.loc[index_row, f'f_app_time_{packagename}'] + time_spent
            df_session_groups.loc[index_row, f'f_app_time_{packagename}'] = pre_time

            # ___________________________________________________________________________________________#
            # save category count
            app_category = ML_helpers.get_app_category(packagename)

            if f'f_app_category_count_{app_category}' not in df_session_groups.columns:
                df_session_groups[f'f_app_category_count_{app_category}'] = 0

            pre_count_cat = df_session_groups.loc[index_row, f'f_app_category_count_{app_category}'] + 1
            df_session_groups.loc[index_row, f'f_app_category_count_{app_category}'] = pre_count_cat

            # ___________________________________________________________________________________________#
            # save time spent on category
            if f'f_app_category_time_{app_category}' not in df_session_groups.columns:
                df_session_groups[f'f_app_category_time_{app_category}'] = 0  # pd.Timedelta(days=0)

            pre_time_cat = df_session_groups.loc[
                                index_row, f'f_app_category_time_{app_category}'] + time_spent
            df_session_groups.loc[index_row, f'f_app_category_time_{app_category}'] = pre_time_cat

            # ___________________________________________________________________________________________#
            currentApp = {"packageName": 'UNKNOWN_first', "timestamp": np.nan}

        # ---------------- Save the seuquence list -------------------#
        if len(current_sequence_list) > 0:
            array = np.array([object])
            array[0] = current_sequence_list
            # print(current_sequence_list)
            df_session_groups.loc[index_row, 'f_sequences'] = array
            current_sequence_list = []

        if len(current_app_sequence_list) > 0:
            array = np.array([object])
            array[0] = current_app_sequence_list
            # print(current_sequence_list)
            df_session_groups.loc[index_row, 'f_sequences_apps'] = array
            current_app_sequence_list = []


        #### after loop over logs: FREQUENCIES APP USAGES ####
        app_category_columns = [col for col in df_session_groups if (col.startswith('f_app_category_count_') and not col.endswith('freq'))]
        for a_app_cat_col in app_category_columns:
            df_session_groups.loc[index_row, f'{a_app_cat_col}_freq'] = \
                df_session_groups.loc[index_row, f'{a_app_cat_col}'] / df_session_groups.loc[index_row, 'f_session_group_length_active'].seconds

        app_category_columns = [col for col in df_session_groups if (col.startswith('f_app_category_time_') and not col.endswith('freq'))]
        for a_app_cat_col in app_category_columns:
            df_session_groups.loc[index_row, f'{a_app_cat_col}_freq'] = \
                df_session_groups.loc[index_row, f'{a_app_cat_col}'] / df_session_groups.loc[
                    index_row, 'f_session_group_length_active'].seconds

        # scroll frequency calculated within app (relative to time spent in app)
        app_category_columns = [col for col in df_session_groups if
                                (col.startswith('f_scrolls_app_category_') and not col.endswith('freq'))]
        for a_app_cat_col in app_category_columns:
            category_name = a_app_cat_col.split("_")[-1]
            if not f'f_app_category_time_{category_name}' in df_session_groups or df_session_groups.loc[index_row, f'f_app_category_time_{category_name}'] == 0:
                continue
                # TODO it occured that scrolls are logged, but no time in app
            df_session_groups.loc[index_row, f'{a_app_cat_col}_freq'] = \
                df_session_groups.loc[index_row, f'{a_app_cat_col}'] / df_session_groups.loc[
                    index_row, f'f_app_category_time_{category_name}']

        # the same with clicks
        app_category_columns = [col for col in df_session_groups if
                                (col.startswith('f_clicks_app_category_') and not col.endswith('freq'))]
        for a_app_cat_col in app_category_columns:
            category_name = a_app_cat_col.split("_")[-1]
            if not f'f_app_category_time_{category_name}' in df_session_groups or df_session_groups.loc[index_row, f'f_app_category_time_{category_name}'] == 0:
                continue
            df_session_groups.loc[index_row, f'{a_app_cat_col}_freq'] = \
                df_session_groups.loc[index_row, f'{a_app_cat_col}'] / df_session_groups.loc[
                    index_row, f'f_app_category_time_{category_name}']

    df_session_groups_sorted = df_session_groups.sort_values(by=['group_id'])  # to "remove gaps", when processing only n groups
    print("finished extracting features")
    return df_session_groups, bag_of_apps_vocab






def get_domain_from_url(url):
    """
    Get the main domain from an url
    :param url: the full url to extract the main domain from
    :return: the main domain
    """
    ext = tldextract.extract(url)
    # ExtractResult(subdomain='forums.news', domain='cnn', suffix='com')
    return f'{ext.subdomain}.{ext.domain}.{ext.suffix}'




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