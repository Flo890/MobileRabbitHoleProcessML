import pandas as pd
import numpy as np
import tldextract


def get_features(df_logs, df_sessions):
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

    # Prepare feature columns
    pd.to_datetime(df_sessions['timestamp_1'])
    pd.to_datetime(df_sessions['timestamp_2'])

    df_sessions['f_hour_of_day'] = np.nan
    df_sessions['f_weekday'] = np.nan
    df_sessions['f_session_length'] = np.nan
    df_sessions['f_clicks'] = 0
    df_sessions['f_scrolls'] = 0

    grouped_logs = df_logs.groupby('session_id')

    # Iterate over sessions
    for name, df_group in grouped_logs:
        if name:  # | (not pd.isna(name):
            # Get the df_session row of the current session to assign the values to
            # print(type(df_group))
            # print(df_group.head())
            df_row = df_sessions[(df_sessions['session_id'].values == name)]
            index_row = df_sessions[(df_sessions['session_id'].values == name)].index.item()

            # Feature session length
            # Feature Time of day
            # Feature day of weel
            df_sessions.loc[index_row, 'f_hour_of_day'] = df_row['timestamp_1'].iloc[0].hour
            # df_row['f_hour_of_day'] = df_row['timestamp_1'].hour

            # df_row['f_weekday'] = df_row['timestamp_1'].dt.dayofweek
            df_sessions.loc[index_row, 'f_weekday'] = df_row['timestamp_1'].iloc[0].dayofweek

            # df_row['f_session_length'] = df_row['timestamp_2'] - df_row['timestamp_1']
            df_sessions.loc[index_row, 'f_session_length'] = df_row['timestamp_2'].iloc[0] - df_row['timestamp_1'].iloc[
                0]

            # TODO session counts?

            # Iterate over all logs of the current session group
            for log in df_group.itertuples():
                # index_log = log.Index
                # print(index_log)
                # print(index, log.eventName, type(log.eventName))
                # Feature session counts
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

                # Feature Scroll and click count
                elif (log.eventName == 'ACCESSIBILITY') | (log.event == 'ACCESSIBILITY_KEYBOARD_INPUT'):
                    if log.event == 'TYPE_VIEW_CLICKED':  # TODO also include context clicked, long clicks?
                        df_sessions.loc[index_row, 'f_clicks'] += 1
                        if f'f_clicks_{log.packageName}' not in df_sessions.columns:
                            df_sessions[f'f_clicks_{log.packageName}'] = 0

                        pre = df_sessions.loc[index_row, f'f_clicks_{log.packageName}'] + 1
                        df_sessions.loc[index_row, f'f_clicks_{log.packageName}'] = pre

                    elif log.event == 'TYPE_VIEW_SCROLLED':
                        df_sessions.loc[index_row,'f_scrolls'] += 1
                        if f'f_scrolls_{log.packageName}' not in df_sessions.columns:
                            df_sessions[f'f_scrolls_{log.packageName}'] = 0

                        pre = df_sessions.loc[index_row, f'f_scrolls_{log.packageName}'] + 1
                        df_sessions.loc[index_row, f'f_scrolls_{log.packageName}'] = pre

                # sequences: apps, websites, notificaitons, calls, sms

                # USed application duration

                # Used application count
                # ACTIVITY_RESUMED

                # Used app category count
                # USed app catgort duration

                # Context
                # used Internet connection -> majority? time for each connection in session?
                # Physical activity

    df_sessions.to_csv(fr'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\featuretest.csv')
    return df_sessions


# def get_app_categories:
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

    df_logs = pd.read_pickle(path_testfile_logs)
    # df_sessions = pd.read_csv(path_testfile_sessions)
    df_sessions = pd.read_pickle(path_testfile_sessions)
    t = df_sessions['SO23BA']
    print(t.head())
    get_features(df_logs=df_logs, df_sessions=t)
