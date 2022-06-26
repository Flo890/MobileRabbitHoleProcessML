import pandas as pd
import numpy as np
import tldextract
import ML_helpers


def checkforactivity(df_logs):
    activity_logs = df_logs[(df_logs['eventName'].values == 'ACTIVITY')]
    print('logs empty activty', activity_logs.empty)
    return not activity_logs.empty



def get_features_for_session(df_logs, df_sessions):
    """
    Extract all features from logs for each user
    :param df_logs: the logs dataframe of one user
    :param df_sessions: the extracted session dataframe of one user
    :return: the sessions and feature dataframe for one user
    """

    # Add demographics features
    path_questionnaire_1 = r'C:\projects\rabbithole\RabbitHoleProcess\data\rawData'
    df_MRH1 = pd.read_csv(f'{path_questionnaire_1}\MRH1.csv', sep=',')
    studyID = df_logs['studyID'].values[0]
    if not df_MRH1[df_MRH1['IM01_01'].values == studyID].empty:
        df_qu_user = df_MRH1[df_MRH1['IM01_01'].values == studyID].index.item()
        df_sessions['f_demographics_gender'] = df_MRH1.loc[df_qu_user, 'SD01']  # where 1: female, 2: male, 3: non-binary, 4: Prefer not to disclose, 5: Other
        df_sessions['f_demographics_age'] = df_MRH1.loc[df_qu_user, 'SD02_01']

        df_sessions['f_absentminded_use'] = df_MRH1.loc[df_qu_user, df_MRH1.columns.str.startswith('AB01')].mean()
        df_sessions['f_general_use'] = df_MRH1.loc[df_qu_user, df_MRH1.columns.str.startswith('AB02')].mean()
        # WLAN name IM06_01
    else:
        df_sessions['f_demographics_gender'] = 0  # where 1: female, 2: male, 3: non-binary, 4: Prefer not to disclose, 5: Other
        df_sessions['f_demographics_age'] = 0
        df_sessions['f_absentminded_use'] = 0
        df_sessions['f_general_use'] = 0

    # Prepare feature columns
    pd.to_datetime(df_sessions['timestamp_1'])
    pd.to_datetime(df_sessions['timestamp_2'])

    df_sessions['f_esm_intention'] = ""
    df_sessions['f_esm_finished_intention'] = ""
    df_sessions['f_esm_more_than_intention'] = ""
    df_sessions['f_esm_track_of_time'] = ""
    df_sessions['f_esm_track_of_space'] = ""
    df_sessions['f_esm_emotion'] = ""
    df_sessions['f_esm_regret'] = ""
    df_sessions['f_esm_agency'] = ""

    df_sessions['f_sequences'] = np.nan
    df_sessions['f_hour_of_day'] = np.nan
    df_sessions['f_weekday'] = np.nan
    df_sessions['f_session_length'] = np.nan
    df_sessions['f_time_since_last_session'] = 0
    df_sessions['f_glances_since_last_session'] = 0
    df_sessions['f_count_session_1h'] = 0
    df_sessions['f_count_session_2h'] = 0
    df_sessions['f_count_session_3h'] = 0
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
    grouped_logs = df_logs.groupby(['studyID', 'session_id'])

    # Iterate over sessions
    for name, df_group in grouped_logs:
        if name[1]:  # | (not pd.isna(name):# if name is not empty
            # Get the df_session row of the current session to assign the values to
            df_row = df_sessions[(df_sessions['count'].values == name[1])]
            index_row = df_sessions[(df_sessions['count'].values == name[1])].index.item()
            current_session_start = df_row['timestamp_1'].iloc[0]

            # ---------------  FEATURE TIME AND GLANCES SINCE LAST SESSION ------------------ #
            index_last_session = index_row - 1
            if index_last_session > 0:
                last_session_end = df_sessions['timestamp_2'].iloc[index_last_session]
                df_sessions.loc[index_row, 'f_time_since_last_session'] = round((current_session_start - last_session_end).total_seconds() * 1000)

                # GLances: get all screen ON_LOCKED events between last session ts and current session ts
                glances_count = len(df_logs[(df_logs['event'].values == 'ON_LOCKED') &
                                            (df_logs['correct_timestamp'].values > last_session_end) &
                                            (df_logs['correct_timestamp'].values < current_session_start)])
                df_sessions.loc[index_row, 'f_glances_since_last_session'] = df_sessions.loc[index_row, 'f_glances_since_last_session'] + glances_count

            # ---------------  FEATURE COUNT SESSION IN LAST HOUR ------------------ #
            # Calculate timestamp one hour ago
            ts_one_hour = current_session_start - pd.Timedelta(hours=1)
            # Get session that started within the last hour
            df_sessions.loc[index_row, 'f_count_session_1h'] = len(df_sessions[(df_sessions['timestamp_1'].values > ts_one_hour) &
                                                                               (df_sessions['timestamp_2'].values < current_session_start)])

            ts_two_hour = current_session_start - pd.Timedelta(hours=2)
            # Get session that started within the last hour
            df_sessions.loc[index_row, 'f_count_session_2h'] = len(df_sessions[(df_sessions['timestamp_1'].values > ts_two_hour) &
                                                                               (df_sessions['timestamp_2'].values < current_session_start)])

            ts_three_hour = current_session_start - pd.Timedelta(hours=3)
            # Get session that started within the last hour
            df_sessions.loc[index_row, 'f_count_session_3h'] = len(df_sessions[(df_sessions['timestamp_1'].values > ts_three_hour) &
                                                                               (df_sessions['timestamp_2'].values < current_session_start)])

            # ---------------  FEATURE HOUR OF DAY ------------------ #
            # df_row['f_hour_of_day'] = df_row['timestamp_1'].hour
            df_sessions.loc[index_row, 'f_hour_of_day'] = df_row['timestamp_1'].iloc[0].hour

            # ---------------  FEATURE WEEKDAY ------------------ #
            # df_row['f_weekday'] = df_row['timestamp_1'].dt.dayofweek
            df_sessions.loc[index_row, 'f_weekday'] = df_row['timestamp_1'].iloc[0].dayofweek

            # ---------------  FEATURE SESSION LENGTH------------------ #
            # df_row['f_session_length'] = df_row['timestamp_2'] - df_row['timestamp_1']
            df_sessions.loc[index_row, 'f_session_length'] = df_row['timestamp_2'].iloc[0] - df_row['timestamp_1'].iloc[0]

            # Iterate over all logs of the current session group
            for log in df_group.itertuples():

                # --------------- FEATURE ESM ------------------ #
                if log.eventName == 'ESM':
                    # emsunlock/unlock intention: description
                    # esmlock: answers: name
                    if log.event == 'ESM_UNLOCK_INTENTION':
                        df_sessions.loc[index_row, 'f_esm_intention'] = log.description

                    elif log.event == 'ESM_LOCK_Q_FINISH':
                        df_sessions.loc[index_row, 'f_esm_finished_intention'] = log.name

                    elif log.event == 'ESM_LOCK_Q_MORE':
                        df_sessions.loc[index_row, 'f_esm_more_than_intention'] = log.name

                    elif log.event == 'ESM_LOCK_Q_TRACK_OF_TIME':
                        df_sessions.loc[index_row, 'f_esm_track_of_time'] = log.name

                    elif log.event == 'ESM_LOCK_Q_TRACK_OF_SPACE':
                        df_sessions.loc[index_row, 'f_esm_track_of_space'] = log.name

                    elif log.event == 'ESM_LOCK_Q_EMOTION':
                        df_sessions.loc[index_row, 'f_esm_emotion'] = log.name

                    elif log.event == 'ESM_LOCK_Q_REGRET':
                        df_sessions.loc[index_row, 'f_esm_regret'] = log.name

                    elif log.event == 'ESM_LOCK_Q_AGENCY':
                        df_sessions.loc[index_row, 'f_esm_agency'] = log.name

                # ---------------  FEATURE SCROLL AND CLICK COUNT ------------------ #
                elif (log.eventName == 'ACCESSIBILITY') | (log.event == 'ACCESSIBILITY_KEYBOARD_INPUT'):

                    if log.event == 'TYPE_VIEW_CLICKED':
                        df_sessions.loc[index_row, 'f_clicks'] += 1
                        if f'f_clicks_{log.packageName}' not in df_sessions.columns:
                            df_sessions[f'f_clicks_{log.packageName}'] = 0

                        pre = df_sessions.loc[index_row, f'f_clicks_{log.packageName}'] + 1
                        df_sessions.loc[index_row, f'f_clicks_{log.packageName}'] = pre

                        # ----- save clicks per app category ------#
                        app_category = ML_helpers.get_app_category(log.packageName)
                        if f'f_clicks_app_category_{app_category}' not in df_sessions.columns:
                            df_sessions[f'f_clicks_app_category_{app_category}'] = 0

                        pre_count_cat = df_sessions.loc[index_row, f'f_clicks_app_category_{app_category}'] + 1
                        df_sessions.loc[index_row, f'f_clicks_app_category_{app_category}'] = pre_count_cat

                    elif log.event == 'TYPE_VIEW_SCROLLED':
                        df_sessions.loc[index_row, 'f_scrolls'] += 1
                        if f'f_scrolls_{log.packageName}' not in df_sessions.columns:
                            df_sessions[f'f_scrolls_{log.packageName}'] = 0

                        pre = df_sessions.loc[index_row, f'f_scrolls_{log.packageName}'] + 1
                        df_sessions.loc[index_row, f'f_scrolls_{log.packageName}'] = pre

                        # ----- save scrolls per app category ------ #
                        app_category = ML_helpers.get_app_category(log.packageName)
                        if f'f_scrolls_app_category_{app_category}' not in df_sessions.columns:
                            df_sessions[f'f_scrolls_app_category_{app_category}'] = 0

                        pre_count_cat = df_sessions.loc[index_row, f'f_scrolls_app_category_{app_category}'] + 1
                        df_sessions.loc[index_row, f'f_scrolls_app_category_{app_category}'] = pre_count_cat

                # ---------------  FEATURE SEQUENCE BrowserURL------------------ #
                elif log.eventName == 'ACCESSIBILITY_BROWSER_URL':
                    # eventName.ACCESSIBILITY_BROWSER_URL,
                    # event = getEventType(event) (accesibility),
                    # description = browserUrl,
                    # name = "BrowserURL",
                    # packageName = packageName
                    seq = ('BROWSER_URL', get_domain_from_url(log.description))
                    current_sequence_list.append(seq)

                    if 'f_browser_url' not in df_sessions.columns:
                        df_sessions['f_browser_url'] = 0

                    pre = df_sessions.loc[index_row, 'f_browser_url'] + 1
                    df_sessions.loc[index_row, 'f_browser_url'] = pre

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
                            (currentInternetState['connectionType'] == new_state['connectionType']) & (currentInternetState['wifiName'] != new_state['wifiName'])):
                        # Calculate time difference between current and new state
                        if currentInternetState['connectionType'] == 'UNKNOWN_first':
                            currentInternetState = new_state

                        else:
                            dif = new_state['timestamp'] - currentInternetState['timestamp']
                            dif = round(dif.total_seconds() * 1000)
                            # add old state to feature table
                            if currentInternetState['connectionType'] == 'CONNECTED_WIFI':
                                df_sessions.loc[index_row, 'f_internet_connected_WIFI'] += dif

                                #  Also add the tme spent in specific wifi network
                                wifi = currentInternetState['wifiName']
                                if f'f_internet_connected_WIFI_{wifi}' not in df_sessions.columns:
                                    df_sessions[f'f_internet_connected_WIFI_{wifi}'] = 0  # pd.Timedelta(days=0)

                                pre = df_sessions.loc[index_row, f'f_internet_connected_WIFI_{wifi}'] + dif
                                df_sessions.loc[index_row, f'f_internet_connected_WIFI_{wifi}'] = pre

                            elif currentInternetState['connectionType'] == 'CONNECTED_MOBILE':
                                df_sessions.loc[index_row, 'f_internet_connected_mobile'] += dif

                            elif currentInternetState['connectionType'] == 'UNKNOWN':
                                df_sessions.loc[index_row, 'f_internet_disconnected'] += dif

                            else:
                                df_sessions.loc[index_row, 'f_internet_else'] += dif
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
                                dif = round((new_mode['timestamp'] - currentRingerMode['timestamp']).total_seconds() * 1000)

                            # add old state to feature table
                            if currentRingerMode['mode'] == 'NORMAL_MODE':
                                df_sessions.loc[index_row, 'f_ringer_mode_normal'] += dif

                            elif currentRingerMode['mode'] == 'VIBRATE_MODE':
                                df_sessions.loc[index_row, 'f_ringer_mode_vibrate'] += dif

                            elif currentRingerMode['mode'] == 'SILENT_MODE':
                                df_sessions.loc[index_row, 'f_ringer_mode_silent'] += dif

                            elif currentRingerMode['mode'] == 'UNKNOWN':
                                df_sessions.loc[index_row, 'f_ringer_mode_unknown'] += dif

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

                            if f'f_app_count_{packagename}' not in df_sessions.columns:
                                df_sessions[f'f_app_count_{packagename}'] = 0

                            pre_count = df_sessions.loc[index_row, f'f_app_count_{packagename}'] + 1
                            df_sessions.loc[index_row, f'f_app_count_{packagename}'] = pre_count

                            # ___________________________________________________________________________________________#
                            # Save time spent on app
                            time_spent = round((log.correct_timestamp - currentApp['timestamp']).total_seconds() * 1000)

                            if f'f_app_time_{packagename}' not in df_sessions.columns:
                                df_sessions[f'f_app_time_{packagename}'] = 0  # pd.Timedelta(days=0)

                            pre_time = df_sessions.loc[index_row, f'f_app_time_{packagename}'] + time_spent
                            df_sessions.loc[index_row, f'f_app_time_{packagename}'] = pre_time

                            # ___________________________________________________________________________________________#
                            # save category count
                            app_category = ML_helpers.get_app_category(packagename)

                            if f'f_app_category_count_{app_category}' not in df_sessions.columns:
                                df_sessions[f'f_app_category_count_{app_category}'] = 0

                            pre_count_cat = df_sessions.loc[index_row, f'f_app_category_count_{app_category}'] + 1
                            df_sessions.loc[index_row, f'f_app_category_count_{app_category}'] = pre_count_cat

                            # ___________________________________________________________________________________________#
                            # save time spent on category
                            if f'f_app_category_time_{app_category}' not in df_sessions.columns:
                                df_sessions[f'f_app_category_time_{app_category}'] = 0  # pd.Timedelta(days=0)

                            pre_time_cat = df_sessions.loc[
                                               index_row, f'f_app_category_time_{app_category}'] + time_spent
                            df_sessions.loc[index_row, f'f_app_category_time_{app_category}'] = pre_time_cat

                            # ___________________________________________________________________________________________#
                            # save new app as current
                            currentApp = {"packageName": log.packageName, "timestamp": log.correct_timestamp}

            # ---------------  save the last internet and app state with the last log ------------------ #
            last_log_time = df_group['correct_timestamp'].iloc[-1]
            last_log_event = df_group['eventName'].iloc[-1]

            # ____________________________ save last internet sate ___________________________________#
            if (currentInternetState['connectionType'] != 'UNKNOWN_first') & (last_log_event != 'INTERNET'):
                dif = round((last_log_time - currentInternetState['timestamp']).total_seconds() * 1000)

                # add old state to feature table
                if currentInternetState['connectionType'] == 'CONNECTED_WIFI':
                    df_sessions.loc[index_row, 'f_internet_connected_WIFI'] += dif

                    #  Also add the tme spent in specific wifi network
                    wifi = currentInternetState['wifiName']
                    if f'f_internet_connected_WIFI_{wifi}' not in df_sessions.columns:
                        df_sessions[f'f_internet_connected_WIFI_{wifi}'] = 0  # pd.Timedelta(days=0)

                    pre = df_sessions.loc[index_row, f'f_internet_connected_WIFI_{wifi}'] + dif
                    df_sessions.loc[index_row, f'f_internet_connected_WIFI_{wifi}'] = pre

                elif currentInternetState['connectionType'] == 'CONNECTED_MOBILE':
                    df_sessions.loc[index_row, 'f_internet_connected_mobile'] += dif

                elif currentInternetState['connectionType'] == 'UNKNOWN':
                    df_sessions.loc[index_row, 'f_internet_disconnected'] += dif
                currentInternetState = {'connectionType': 'UNKNOWN_first', 'wifiState': 'UNKNOWN_first', 'wifiName': '', 'timestamp': np.nan}

            # ____________________________ save last ringer mode state ___________________________________#
            if (currentRingerMode['mode'] != 'UNKNOWN_first') & (last_log_event != 'RINGER_MODE'):
                #  print(last_log_time, currentRingerMode['timestamp'], current_session_start,  current_session_start > currentRingerMode['timestamp'])
                # Check if cached ringer mode,i if yes take session start if not take chached timestamp
                if current_session_start > currentRingerMode['timestamp']:
                    dif = round((last_log_time - current_session_start).total_seconds() * 1000)
                else:
                    dif = round((last_log_time - currentRingerMode['timestamp']).total_seconds() * 1000)
                # add old state to feature table
                if currentRingerMode['mode'] == 'NORMAL_MODE':
                    df_sessions.loc[index_row, 'f_ringer_mode_normal'] += dif

                elif currentRingerMode['mode'] == 'VIBRATE_MODE':
                    df_sessions.loc[index_row, 'f_ringer_mode_vibrate'] += dif

                elif currentRingerMode['mode'] == 'SILENT_MODE':
                    df_sessions.loc[index_row, 'f_ringer_mode_silent'] += dif

                elif currentRingerMode['mode'] == 'UNKNOWN':
                    df_sessions.loc[index_row, 'f_ringer_mode_unknown'] += dif

                # Do not reset the cached ringer mode, as it only tiggered in change

            # _____________________________ save last app state _____________________________________#
            if (currentApp['packageName'] != 'UNKNOWN_first') & (last_log_event != 'USAGE_EVENTS'):
                packagename = currentApp['packageName']

                if f'f_app_count_{packagename}' not in df_sessions.columns:
                    df_sessions[f'f_app_count_{packagename}'] = 0

                pre_count = df_sessions.loc[index_row, f'f_app_count_{packagename}'] + 1
                df_sessions.loc[index_row, f'f_app_count_{packagename}'] = pre_count

                # ___________________________________________________________________________________________#
                # Save time spent on app
                time_spent = round((last_log_time - currentApp['timestamp']).total_seconds() * 1000)

                if f'f_app_time_{packagename}' not in df_sessions.columns:
                    df_sessions[f'f_app_time_{packagename}'] = 0  # pd.Timedelta(days=0)

                pre_time = df_sessions.loc[index_row, f'f_app_time_{packagename}'] + time_spent
                df_sessions.loc[index_row, f'f_app_time_{packagename}'] = pre_time

                # ___________________________________________________________________________________________#
                # save category count
                app_category = ML_helpers.get_app_category(packagename)

                if f'f_app_category_count_{app_category}' not in df_sessions.columns:
                    df_sessions[f'f_app_category_count_{app_category}'] = 0

                pre_count_cat = df_sessions.loc[index_row, f'f_app_category_count_{app_category}'] + 1
                df_sessions.loc[index_row, f'f_app_category_count_{app_category}'] = pre_count_cat

                # ___________________________________________________________________________________________#
                # save time spent on category
                if f'f_app_category_time_{app_category}' not in df_sessions.columns:
                    df_sessions[f'f_app_category_time_{app_category}'] = 0  # pd.Timedelta(days=0)

                pre_time_cat = df_sessions.loc[
                                   index_row, f'f_app_category_time_{app_category}'] + time_spent
                df_sessions.loc[index_row, f'f_app_category_time_{app_category}'] = pre_time_cat

                # ___________________________________________________________________________________________#
                currentApp = {"packageName": 'UNKNOWN_first', "timestamp": np.nan}

            # ---------------- Save the seuquence list -------------------#
            if len(current_sequence_list) > 0:
                array = np.array([object])
                array[0] = current_sequence_list
                # print(current_sequence_list)
                df_sessions.loc[index_row, 'f_sequences'] = array
                current_sequence_list = []

            if len(current_app_sequence_list) > 0:
                array = np.array([object])
                array[0] = current_app_sequence_list
                # print(current_sequence_list)
                df_sessions.loc[index_row, 'f_sequences_apps'] = array
                current_app_sequence_list = []

    print("finished extracting features")
    return df_sessions, bag_of_apps_vocab


def get_domain_from_url(url):
    """
    Get the main domain from an url
    :param url: the full url to extract the main domain from
    :return: the main domain
    """
    ext = tldextract.extract(url)
    # ExtractResult(subdomain='forums.news', domain='cnn', suffix='com')
    return f'{ext.subdomain}.{ext.domain}.{ext.suffix}'
