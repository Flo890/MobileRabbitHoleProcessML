import pandas as pd

def build_session_sessions(df, threshold_sec):
    """

    :param df: sessions of ONE user, in chronological order
    :param threshold_seconds: how many seconds may max. be between to sessions, in order to group them together
    :return: list of session-sessions, each consisting of a list dfs (= usages that belong to the session-session)
    """
    print('Grouping sessions to session groups')
    session_sessions = []

    # no usage sessions
    if len(df) == 0:
        return session_sessions

    session_sessions.append({'dfs':[df.iloc[[0]]]})  # create first session-session with first usage
    for i in range(1, len(df)):
        current_usage = df.iloc[[i]]
        # get last usage of last session-session
        last_usage = session_sessions[-1]['dfs'][-1]
        # check if the current one is close enough
        gap_duration = current_usage['timestamp_1'][i] - last_usage['timestamp_2'][i-1]
        if gap_duration.seconds < threshold_sec:
            session_sessions[-1]['dfs'].append(current_usage)
        else:
            session_sessions.append({'dfs': [current_usage]})

    return session_sessions


def session_sessions_to_aggregate_df(sessions):
    """
    takes the list of sessions from build_session_sessions and creates a dataframe, listing those groups and the according session ids
    :param sessions:
    :return:
    """
    print('Creating aggregated session-group dataframe')
    session_ids = []
    group_ids = []
    timestamps_one = []
    timestamps_two = []
    count = []
    counter = 0
    for i in range(len(sessions)):
        # session ids
        usage_ids = []
        for j in range(len(sessions[i]['dfs'])):
            usage_ids.append(sessions[i]['dfs'][j]['session_id'])
        session_ids.append(usage_ids)
        # group id (the new session id)
        group_id = usage_ids[0][counter].split("-")[1]
        # timestamps for group
        timestamps_one.append(sessions[i]['dfs'][0]['timestamp_1'][counter])
        timestamps_two.append(sessions[i]['dfs'][-1]['timestamp_2'][counter+len(sessions[i]['dfs'])-1])
        # count
        count.append(counter+1)

        for usage_id in usage_ids:
            group_id += ("-"+usage_id[counter].split("-")[0])
            counter += 1
        group_ids.append(group_id)

    df = pd.DataFrame({'group_id': group_ids, 'count': count, 'session_ids': session_ids, 'timestamp_1': timestamps_one, 'timestamp_2': timestamps_two})
    return df

if __name__ == '__main__':
    df = pd.read_pickle(
        "C:\\projects\\rabbithole\\RabbitHoleProcess\\data\\dataframes\\sessions_with_features\\AN23GE.pickle")
    res = build_session_sessions(df, 120)

    res2 = session_sessions_to_aggregate_df(res)