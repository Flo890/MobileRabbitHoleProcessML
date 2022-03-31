import pandas as pd
import pathlib
import pickle


# Do this all for one user at a time
def process_timestamps(df_logs):
    """
    Converts the timestamps and timezone offsets to a new correct_timestamp colum an orders the df
    :param df_logs: the dataframe of logs for !one! user
    :return: the new ordered logs dataframe
    """
    print("_process timestamps_")

    # Add timezoneOffset to timestamps
    #  df_logs["correct_timestamp"] = df_logs["timestamp"] + df_logs["timezoneOffset"]

    # Add timezoneOffset to timestamps and convert milliseconds to date format
    df_logs['correct_timestamp'] = pd.to_datetime(df_logs["timestamp"] + df_logs["timezoneOffset"], yearfirst=True, unit='ms')

    # TODO Drop the old timestamp and timezone offset?

    # order rows to logging timestamp
    df_logs.sort_values(by=['correct_timestamp'], ascending=True, inplace=True)

    # Reset index?
    # df_logs_ordered = df_logs.reset_index().set_index(df_logs['correct_timestamp'], drop=False)

    # add further interesting variables etc weekday
    # For weekday as number: 0: is Monday and  6 is Sunday
    df_logs['weekday'] = df_logs['correct_timestamp'].dt.dayofweek

    # For weekday as word: 'Wednesday'
    # df_logs_ordered['weekday'] = df_logs_ordered['correct_timestamp'].strftime('%A')

    return df_logs


if __name__ == '__main__':
    dataframe_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\test'
    pathlist = pathlib.Path(dataframe_dir).glob('**/*.pickle')

    for data_path in pathlist:
        path_in_str = str(data_path)
        df_logs = pd.read_pickle(path_in_str)
        # print(df_logs.columns.values)
        # print("______________________")

        df_logs_t = process_timestamps(df_logs)  # can i assign to same variable again?
        with open(fr'{dataframe_dir}\{data_path.stem}.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(df_logs_t, f, pickle.HIGHEST_PROTOCOL)

        # print(df_logs_t.head())
        # print(df_logs_t.columns.values)
