import pandas as pd
import numpy as np
import tldextract


def get_features(df_logs, df_sessions):
    """
    Extract all features from logs for each user
    :param df_logs: the logs dataframe of one user
    :param df_sessions: the extracted session dataframe of one user
    :return: the sessions and feature dataframe for one user
    """
    study_id = df_sessions['session_id'][0]
    print(f'Extract features for {study_id}')
    # Loop over session list  ,session_id, session_length, timestamp_1, timestamp_2

    # index description event eventName id timestamp timezoneOffset name packageName studyID correct_timestamp weekday
    # dataKey infoText interaction priority subText mobile_BYTES_RECEIVED mobile_BYTES_TRANSMITTED wifi_BYTES_RECEIVED wifi_BYTES_TRANSMITTED category
    # session_id

    # assign to itertuipels df
    # i = row[0] or row.Index
    # df.loc[i, 'column_name'] = some value

    # Group the session with the previously assigened sessionids
    # df_logs.drop((df_logs['session_id'] == ''), inplace=True)
    # print(df_logs.head())

    categories = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\categories\app_categories.csv'
    df_categories = pd.read_csv(categories)

    # Prepare feature columns
    pd.to_datetime(df_sessions['timestamp_1'])
    pd.to_datetime(df_sessions['timestamp_2'])

    df_sessions['f_hour_of_day'] = np.nan
    df_sessions['f_weekday'] = np.nan
    df_sessions['f_session_length'] = np.nan
    df_sessions['f_clicks'] = 0
    df_sessions['f_scrolls'] = 0
    df_sessions['f_internet_connected_WIFI'] = 0
    df_sessions['f_internet_connected_mobile'] = 0
    df_sessions['f_internet_disconnected'] = 0

    currentInternetState = {'connectionType': 'UNKNOWN', 'wifiState': 'UNKNOWN', 'wifiName': '', 'timestamp': np.nan}
    currentApp = {"packageName": 'unknown', "timestamp": np.nan}

    grouped_logs = df_logs.groupby('session_id')

    # Iterate over sessions
    for name, df_group in grouped_logs:
        if name:  # | (not pd.isna(name):# if name is not empty
            # Get the df_session row of the current session to assign the values to
            df_row = df_sessions[(df_sessions['session_id'].values == name)]
            index_row = df_sessions[(df_sessions['session_id'].values == name)].index.item()

            current_sequence_list = []

            ######## FEATURE HOUR OF DAY ########
            # df_row['f_hour_of_day'] = df_row['timestamp_1'].hour
            df_sessions.loc[index_row, 'f_hour_of_day'] = df_row['timestamp_1'].iloc[0].hour

            ######## FEATURE WEEKDAY ########
            # df_row['f_weekday'] = df_row['timestamp_1'].dt.dayofweek
            df_sessions.loc[index_row, 'f_weekday'] = df_row['timestamp_1'].iloc[0].dayofweek

            ######## FEATURE SESSION LENGTH ########
            # df_row['f_session_length'] = df_row['timestamp_2'] - df_row['timestamp_1']
            df_sessions.loc[index_row, 'f_session_length'] = df_row['timestamp_2'].iloc[0] - df_row['timestamp_1'].iloc[
                0]

            # TODO session counts?
            ######## FEATURE SESSION COUNTs ########

            # Iterate over all logs of the current session group
            for log in df_group.itertuples():
                # index_log = log.Index
                # print(index_log)
                # print(index, log.eventName, type(log.eventName))

                ######## FEATURE ESM ########
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

                ######## FEATURE SCROLL AND CLICK COUNT ########
                elif (log.eventName == 'ACCESSIBILITY') | (log.event == 'ACCESSIBILITY_KEYBOARD_INPUT'):
                    if log.event == 'TYPE_VIEW_CLICKED':  # TODO also include context clicked, long clicks?
                        df_sessions.loc[index_row, 'f_clicks'] += 1
                        if f'f_clicks_{log.packageName}' not in df_sessions.columns:
                            df_sessions[f'f_clicks_{log.packageName}'] = 0

                        pre = df_sessions.loc[index_row, f'f_clicks_{log.packageName}'] + 1
                        df_sessions.loc[index_row, f'f_clicks_{log.packageName}'] = pre

                    elif log.event == 'TYPE_VIEW_SCROLLED':
                        df_sessions.loc[index_row, 'f_scrolls'] += 1
                        if f'f_scrolls_{log.packageName}' not in df_sessions.columns:
                            df_sessions[f'f_scrolls_{log.packageName}'] = 0

                        pre = df_sessions.loc[index_row, f'f_scrolls_{log.packageName}'] + 1
                        df_sessions.loc[index_row, f'f_scrolls_{log.packageName}'] = pre

                ######## FEATURE BrowserURL ########
                elif log.eventName == 'ACCESSIBILITY_BROWSER_URL':
                    current_sequence_list.append(log)

                ######## FEATURE NOTIFICATION ########
                elif log.eventName == 'NOTIFICATION':
                    # TODO CLEANUP, meybe use usage stats for that as well
                    current_sequence_list.append(log)

                ######## FEATURE INTERNET CONNECTION ########
                elif log.eventName == 'INTERNET':
                    # description: Type of connection CONNECTED_WIFI, Event: Disabled, enabled, name: Wifi Name
                    # currentstate: triple of connectiontype, state, wifi name and start timestamp of state
                    new_state = {'connectionType': log.description,
                                 'wifiState': log.event,
                                 'wifiName': log.name,
                                 'timestamp': log.correct_timestamp
                                 }

                    if not currentInternetState['connectionType'] == new_state['connectionType']:
                        # Calculate time difference between current and new state
                        dif = new_state['timestamp'] - currentInternetState['timestamp']
                        # Make difference for each connectionType

                        # TODO check also if wifi Name changes not just state?

                        # add old state to feature table
                        df_sessions.loc[index_row, 'f_scrolls'] += 1
                        if currentInternetState['connectionType'] == 'CONNECTED_WIFI':
                            df_sessions.loc[index_row, 'f_internet_connected_WIFI'] += dif

                            #  Also add the tme spent in specific wifi network
                            wifi = currentInternetState['wifiName']
                            if f'f_internet_connected_WIFI_{wifi}' not in df_sessions.columns:
                                df_sessions[f'f_internet_connected_WIFI_{wifi}'] = 0

                            pre = df_sessions.loc[index_row, f'f_internet_connected_WIFI_{wifi}'] + dif
                            df_sessions.loc[index_row, f'f_internet_connected_WIFI_{wifi}'] = pre

                        elif currentInternetState['connectionType'] == 'CONNECTED_MOBILE':
                            df_sessions.loc[index_row, 'f_internet_connected_mobile'] += dif

                        elif currentInternetState['connectionType'] == 'UNKNOWN':
                            df_sessions.loc[index_row, 'f_internet_disconnected'] += dif

                        # TODO elif currentInternetState['connectionType'] == 'CONNECTED_ETHERNET':
                        # TODO elif currentInternetState['connectionType'] == 'CONNECTED_VPN':

                        # Save the new state as current state
                        currentInternetState = new_state

                elif log.eventName == 'PHONE':
                    current_sequence_list.append(log)

                elif log.eventName == 'SMS':
                    current_sequence_list.append(log)

                elif log.eventName == 'ACTIVITY':
                    print("found acitvity")

                ######## FEATURE USED APPS ########
                # TODO Drop unused usage events before iterrating?
                elif log.eventName == 'USAGE_EVENTS':
                    # description: applicationname, event: Usage event type, name: classname, packageName: packagename
                    #
                    # if (log.event == 'ACTIVITY_RESUMED') | (log.event == 'MOVE_TO_FOREGROUND'):
                    # elif (log.event == 'ACTIVITY_PAUSED') | (log.event == 'MOVE_TO_BACKGROUND'):
                    # elif log.event == 'ACTIVITY_STOPPED':

                    # check the package name if different from before
                    if not currentApp['packageName'] == log.packageName:
                        if log.event == 'ACTIVITY_RESUMED':
                            # Save the new app to sequence list
                            current_sequence_list.append(log)
                            # Save current app count and time spent on it
                            time_spent = log.correct_timestamp - currentApp['timestamp']
                            packagename = currentApp['packageName']

                            if f'f_app_count_{packagename}' not in df_sessions.columns:
                                df_sessions[f'f_app_count_{packagename}'] = 0

                            pre_count = df_sessions.loc[index_row, f'f_app_count_{packagename}'] + 1
                            df_sessions.loc[index_row, f'f_app_count_{packagename}'] = pre_count

                            if f'f_app_time_{packagename}' not in df_sessions.columns:
                                df_sessions[f'f_app_time_{packagename}'] = 0

                            pre_time = df_sessions.loc[index_row, f'f_app_time_{packagename}'] + time_spent
                            df_sessions.loc[index_row, f'f_app_time_{packagename}'] = pre_time

                            # save new app as current
                            currentApp = {"packageName": log.packageName, "timestamp": log.correct_timestamp}

                    # Ignore other events that activity resumed
                    # Check if it is activity resumed (ignore stopped and paused) when package name different and take this as endpoint for previous

                    # count and add colum if no existing yet
                    if f'f_internet_connected_WIFI_{wifi}' not in df_sessions.columns:
                        df_sessions[f'f_internet_connected_WIFI_{wifi}'] = 0

                    pre = df_sessions.loc[index_row, f'f_internet_connected_WIFI_{wifi}'] + dif
                    df_sessions.loc[index_row, f'f_internet_connected_WIFI_{wifi}'] = pre

            ######## FEATURE SEQUENCES ########
            # Save sequence list to feature tabel
            df_sessions['f_sequences'] = current_sequence_list

            # sequences: apps, websites, notificaitons, calls, sms

            # USed application duration

            # Used application count
            # ACTIVITY_RESUMED for unbder andorid 29 MOVE TO FOREGROUND

            # Used app category count
            # USed app catgort duration

            # Context
            # used Internet connection -> majority? time for each connection in session?
            # Physical activity

            # sequences: apps, websites, notificaitons, calls, sms

    df_sessions.to_csv(fr'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\featuretest.csv')
    return df_sessions


def get_app_category(df_categories, packagename):
    category = df_categories[(df_categories['apps.packageName'].values == packagename)]['category.new']
    if not category.empty:
        return category.values[0]
    else:
        return 'UNKNOWN'


def get_domain_from_url(url):
    """
    Get the main domain from an url
    :param url: the full url to extract the main domain from
    :return: the main domain
    """
    ext = tldextract.extract(url)
    # ExtractResult(subdomain='forums.news', domain='cnn', suffix='com')
    # print(ext.domain, ext.subdomain, ext.suffix, ext.registered_domain)
    return f'{ext.subdomain}.{ext.domain}.{ext.suffix}'


if __name__ == '__main__':
    path_testfile_sessions = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\user-sessions.pickle'
    path_testfile_logs = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\testing\SO23BA.pickle'

    # Show all colums when printing head()
    pd.set_option('display.max_columns', None)

    # test_sessions = pd.read_csv(path_testfile_sessions)
    # logs = pd.read_pickle(path_testfile_logs)

    # get_features(logs, test_sessions)
    # get_domain_from_url('https://developer.android.com/reference/android/app/usage/UsageEvents.Event')

    df_all_logs = pd.read_pickle(path_testfile_logs)
    # df_sessions = pd.read_csv(path_testfile_sessions)
    df_all_sessions = pd.read_pickle(path_testfile_sessions)
    t = df_all_sessions['SO23BA']
    print(t.head())
    get_features(df_logs=df_all_logs, df_sessions=t)
