import pandas as pd
import pathlib
import pickle

import grouped_sessions
import featureExtractionGroup





if __name__ == '__main__':

    path_list = pathlib.Path(r'D:\usersorted_logs_preprocessed').glob('**/*.pickle')

    for data_path in path_list:
        print(f'###### {data_path.stem} ######')
        path_in_str = str(data_path)
        # df_all_logs = pd.read_pickle(path_in_str)
        print('session_file', f'C:\\projects\\rabbithole\\RabbitHoleProcess\\data\\dataframes\\sessions_raw_extracted\\{data_path.stem}-sessions.pickle')
        df_all_sessions = pd.read_pickle(f'C:\\projects\\rabbithole\\RabbitHoleProcess\\data\\dataframes\\sessions_raw_extracted\\{data_path.stem}-sessions.pickle')

        print(data_path.stem)
        # session_for_id = df_all_sessions[data_path.stem]


        df = pd.read_pickle(
            f"C:\\projects\\rabbithole\\RabbitHoleProcess\\data\\dataframes\\sessions_with_features\\{data_path.stem}.pickle")
        res = grouped_sessions.build_session_sessions(df, 120)

        res2 = grouped_sessions.session_sessions_to_aggregate_df(res)

        df_logs = pd.read_pickle(f"D:\\usersorted_logs_preprocessed\\{data_path.stem}.pickle")
        #"C:\\Users\\florianb\\Downloads\\AN23GE.pickle")
              # AN23GE.pickle")   AN09BI    LE13FO

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
            f"C:\\projects\\rabbithole\\RabbitHoleProcess\\data\\dataframes\\sessions_raw_extracted\\{data_path.stem}-sessions.pickle")

        df_session_group_features = featureExtractionGroup.get_features_for_sessiongroup(df_logs, df_all_sessions, res2)

        print('Writing out-file')
        with open(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\session_group_features\{data_path.stem}.pickle', 'wb') as f:
            pickle.dump(df_session_group_features[0], f, pickle.HIGHEST_PROTOCOL)