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
    # Add timezoneOffset to timestamps and convert milliseconds to date format
    df_logs['correct_timestamp'] = pd.to_datetime(df_logs["timestamp"] + df_logs["timezoneOffset"], yearfirst=True, unit='ms')

    # order rows to logging timestamp
    df_logs.sort_values(by=['correct_timestamp'], ascending=True, inplace=True)

    return df_logs


if __name__ == '__main__':
    dataframe_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\test'
    pathlist = pathlib.Path(dataframe_dir).glob('**/*.pickle')

    for data_path in pathlist:
        path_in_str = str(data_path)
        df_logs_all = pd.read_pickle(path_in_str)

        df_logs_t = process_timestamps(df_logs_all)  # can i assign to same variable again?
        with open(fr'{dataframe_dir}\{data_path.stem}.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(df_logs_t, f, pickle.HIGHEST_PROTOCOL)

