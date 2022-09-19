import numpy
import pandas as pd
import pathlib
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

import grouped_sessions
import featureExtractionGroup

import datetime




def concat_sessions():
    print('concat sessions')
    path_list = pathlib.Path(rf'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\session_group_features').glob('**/*.pickle')
    sessions = []

    for data_path in path_list:
        path_in_str = str(data_path)
        sessions.append(pd.read_pickle(path_in_str))

    df_concat = pd.concat(sessions, ignore_index=False)
    df_concat.reset_index(drop=True, inplace=True)

    df_concat.to_csv(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\user-session-group_features_all.csv')
    with open(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\user-session-group_features_all.pickle', 'wb') as f:
        pickle.dump(df_concat, f, pickle.HIGHEST_PROTOCOL)


def bag_of_apps_create_vocab():
    """
    Create the vocabulary for the bag of apps method
    :return:
    """
    print('create bag of apps vocabulary')
    # get the vocabulary, which is every application package used by every user?
    # create vocab from every users app list
    path_list = pathlib.Path(r'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\bag_of_apps_vocabs').glob('**/*.pickle')
    vocab = []
    for data_path in path_list:
        path_in_str = str(data_path)
        print(path_in_str)
        user_vocab = pd.read_pickle(path_in_str)
        vocab = list(set(vocab + user_vocab))

    print(len(vocab))
    print(vocab)
    with open(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\bag_of_apps_vocab.pickle', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


def bag_of_apps_create_bags():
    """
    Create a bag of apps vector for each session in the df of all sessions
    :return:
    """
    print('create bag of apps')
    # go over every session and create vector of app count
    vocab = pd.read_pickle(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\bag_of_apps_vocab.pickle')
    df_sessions = pd.read_pickle(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\user-session-group_features_all.pickle')

    df_sessions['f_bag_of_apps'] = pd.Series(dtype='object')
    df_sessions['f_sequences_apps'].fillna(" ", inplace=True)

    for index, session in enumerate(df_sessions.itertuples(index=False)):
        bag_vector = [0] * len(vocab)

        app_list = session.f_sequences_apps[0]
        for a in app_list:
            for i, app in enumerate(vocab):
                if app == a:
                    bag_vector[i] += 1

        array = np.array([object])
        array[0] = bag_vector
        df_sessions.at[index, 'f_bag_of_apps'] = bag_vector

    with open(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\user-session-group_features_all.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def drop_sequences():
    """
    Drop the sequences features
    :return:
    """
    print('drop sequences')
    df_sessions = pd.read_pickle(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\user-session-group_features_all.pickle')
    df_sessions.drop(columns=['f_sequences'], inplace=True)
    # df_sessions.to_csv(fr'{dataframe_dir_users}\user-sessions_features_all_noseq.csv')
    with open(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\user-session-group_features_all.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def convert_timedeletas():
    """
    Convert time deltas to seconds
    :return:
    """
    print('convert timedeltas')
    df_sessions = pd.read_pickle(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\user-session-group_features_all.pickle')

    df_sessions.dropna(subset=['f_session_group_timespan'], inplace=True)
    df_sessions.reset_index(drop=True, inplace=True)

    if isinstance(df_sessions['f_session_group_timespan'], datetime.timedelta):
        df_sessions['f_session_group_timespan'] = df_sessions['f_session_group_timespan'].apply(lambda x: round(x.total_seconds() * 1000))
        df_sessions['f_session_group_length_active'] = df_sessions['f_session_group_length_active'].apply(lambda x: round(x.total_seconds() * 1000))
        df_sessions['f_session_group_length_active_mean'] = df_sessions['f_session_group_length_active_mean'].apply(lambda x: round(x.total_seconds() * 1000))
        df_sessions['f_session_group_length_active_median'] = df_sessions['f_session_group_length_active_median'].apply(lambda x: round(x.total_seconds() * 1000))
        df_sessions['f_session_group_length_active_sd'] = df_sessions['f_session_group_length_active_sd'].apply(lambda x: numpy.nan if pd.isnull(x) else round(x.total_seconds() * 1000))


    with open(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\user-session-group_features_all.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)



def one_hot_encoding_scilearn():
    """
    One-hot-encode features using scilearn OneHotEncoder
    :return:
    """
    print('on hot encoding scilearn')
    df_sessions = pd.read_pickle(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\user-session-group_features_all.pickle')
   # df_sessions = df_sessions.replace(r'^\s*$', np.nan, regex=True)
    enc = OneHotEncoder(handle_unknown='ignore')

    to_encode = ['f_demographics_gender',
              #   'f_esm_finished_intention',
                 'f_esm_atleastone_more_than_intention',
               #  'f_esm_emotion',
            #     'f_esm_track_of_time',
            #     'f_esm_track_of_space',
                 'f_esm_atleastone_regret',
             #    'f_esm_agency',
                 'f_hour_of_day',
                 'f_weekday']

    for column in to_encode:
        # passing bridge-types-cat column (label encoded values of bridge_types)
        if 'f_demographics_gender' in df_sessions:
            end = enc.fit_transform(df_sessions[[column]]).toarray()
            column_name = enc.get_feature_names_out([column])
            enc_df = pd.DataFrame(end, columns=column_name)

            # merge with main df bridge_df on key values
            # df_sessions = df_sessions.join(enc_df)
            df_sessions = pd.concat([df_sessions, enc_df], axis=1).drop(columns=[column])
        else:
            print('Skipping encoding for column '+column+'; Column doesnt exist')

    # df_sessions.to_csv(fr'{dataframe_dir_users}\user-sessions_features_encoded_sci.csv')

    with open(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\user-session-group_features_all.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def filter_sessions_outliners_all():
    """
    Filter and remove the outliner sessions
    :return:
    """

    feature_path = fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\user-session-group_features_all.pickle'
    df_all_sessions = pd.read_pickle(feature_path)

    # sns.distplot(df_session_features['f_session_length'] / pd.Timedelta(milliseconds=1))
    # plt.show()
    sns.boxplot(df_all_sessions['f_session_group_timespan'])
    plt.show()

    # quantile LIMIT -------------------------------------- # TODO limits checken
    upper_limit = df_all_sessions['f_session_group_timespan'].quantile(0.996)  # bis 5.5h bei 0.995 bis 3.5h
    lower_limit = df_all_sessions['f_session_group_timespan'].quantile(0.01)

    df_sessions_filtered = df_all_sessions[(df_all_sessions['f_session_group_timespan'] <= upper_limit) & (df_all_sessions['f_session_group_timespan'] >= lower_limit)]
    print(type(df_sessions_filtered))
    sns.boxplot(df_sessions_filtered['f_session_group_timespan'])
    plt.show()

    print(upper_limit)
    print(lower_limit)

    with open(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\user-session-group_features_all.pickle', 'wb') as f:
        pickle.dump(df_sessions_filtered, f, pickle.HIGHEST_PROTOCOL)


def filter_users():
    """
    Filter the sessions of user to only include participants that completed both questionnaires
    :return:
    """
    print('filter users')
    df_sessions = pd.read_pickle(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\user-session-group_features_all.pickle')
    df_MRH1_raw = pd.read_csv(fr'C:\projects\rabbithole\RabbitHoleProcess\data\rawData\MRH1.csv', sep=',')
    df_MRH2 = pd.read_csv(fr'C:\projects\rabbithole\RabbitHoleProcess\data\rawData\MRH2.csv', sep=',')

    studIDs_mrh2 = df_MRH2['IM01_01'].apply(lambda x: x.upper())

    # Find the users to drop
    # to_drop = []
    # grouped_logs = df_sessions.groupby(['studyID'])
    # # Iterate over sessions
    # for name, df_group in grouped_logs:
    #     print(name)
    #     if name.upper() not in studIDs_mrh2.values:
    #         to_drop.append(name)

    df_sessions['studyID'] = df_sessions['group_id'].apply(lambda row: row.split("-")[0])

    to_drop = ['AN09BI', 'NI23HE', 'PI02MA']

    print(to_drop)
    for study_id in to_drop:
        df_sessions.drop(df_sessions.index[df_sessions['studyID'] == study_id], inplace=True)

    # df_sessions.to_csv(fr'{dataframe_dir_ml}\sessions_filtered_users.csv')
    with open(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\user-session-group_features_all.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def reduce_feature_dimension():
    """
    Reduce the feature dimension by summarizing app features that only occurred under a threshold value in a feature_other
    :return:
    """
    print("reduce_feature_dimension")
    # Create colum with other for each type
    # Get all columns starting with feature id
    # Count occurences of 1
    # if under threshhold (5?) add to other categorie

    # f_clicks_app_category
    # f_scrolls_app_category_
    # f_scrolls_ package
    # f_clicks__packge
    # f_app_count
    # f_app_time
    # f_app_category_count_
    # f_app_category_time

    # repeat for all types
    threshold = 5
    feature_to_reduce = ['f_clicks_app_category',
                         'f_scrolls_app_category',
                         'f_scrolls',
                         'f_clicks',
                         'f_app_count',
                         'f_app_time',
                         'f_app_category_count',
                         'f_app_category_time']

    df_sessions = pd.read_pickle(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\user-session-group_features_all.pickle')
    df_sessions.reset_index(drop=True, inplace=True)
    # df_sessions.fillna(0.0, inplace=True)
    # df_sessions.drop(columns = ['session_length', 'timestamp_1', 'timestamp_2', 'f_sequences_apps'])

    for item in feature_to_reduce:

        colums_f = [x for x in df_sessions.columns.values if x.startswith(item)]
        other_colum_name = f'{item}_other'
        print(other_colum_name)
        df_sessions[other_colum_name] = 0

        for column_name in colums_f:
            df_sessions[column_name] = df_sessions[column_name].fillna(0)

            count = (df_sessions[column_name] > 0).sum()

            if count <= threshold:
                df_sessions[other_colum_name] = df_sessions[other_colum_name] + df_sessions[column_name]
                df_sessions.drop(columns=[column_name], inplace=True)

                # old = df_sessions.loc[df_sessions[column] > 0, other_colum_name]
                # print(df_sessions[df_sessions[column] > 0, other_colum_name])
                # print(old)
                # dfs = df_sessions[df_sessions[column_name] > 0]
                # for index, row in enumerate(df_sessions[df_sessions[column_name] > 0].itertuples()):
                #     #print(row, type(row))
                #     #print(row[column_name][0], row.other_colum_name)
                #     #print( type(row[column]), row[column])
                #     #new = row[column_name] + row[other_colum_name]
                #     print(index)
                #     print(df_sessions.loc[index, column_name], df_sessions.loc[index, other_colum_name])
                #     pre = df_sessions.loc[index, other_colum_name] + df_sessions.loc[index, column_name]
                #     df_sessions.loc[index, other_colum_name] = pre
                #     print("new", df_sessions.loc[index, other_colum_name])

                # df_sessions.drop(columns=[column], inplace=True)

                # print(df_sessions[df_sessions[column] > 0, other_colum_name])
                # df_sessions.at[[df_sessions[column] == 1], other_colum_name] += count
                # print(t)
                # print(t.columns)
                # df_sessions.drop(columns=['f_sequences'], inplace=True)

    df_sessions.to_csv(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\user-session-group_features_all_f_reduction.csv')
    with open(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\user-session-group_features_all_f_reduction.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def create_labels_single():
    """
    Label the session with the target class, using a single ES features as target
    """
    print('create label')

    df_sessions = pd.read_pickle(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\user-session-group_features_all.pickle')
    df_sessions.reset_index(drop=True, inplace=True)
    rh = 'rabbit_hole'
    no_rh = 'no_rabbithole'

    df_sessions.insert(6, 'target_label', '')

    for index, session in enumerate(df_sessions.itertuples(index=False)):

        # ['f_esm_atleastone_regret', 'f_esm_atleastone_more_than_intention']
        if df_sessions.loc[index, 'f_esm_atleastone_more_than_intention_Yes'] == 1:
            df_sessions.at[index, 'target_label'] = rh
        else:
            df_sessions.at[index, 'target_label'] = no_rh

    # df_sessions = drop_esm_features(df_sessions)

    with open(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\labled_data\sessions_features_labeled_more_than_intention_with_esm.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def remove_personalised_features():
    """
    Remove the personalized feature to prepare the session for machine learning
    :return:
    """
    path = fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\labled_data\sessions_features_labeled_more_than_intention_with_esm.pickle'
    df_sessions = pd.read_pickle(path)

    df_sessions.drop(df_sessions.index[df_sessions['f_session_group_timespan'].isnull()], inplace=True)

    df_sessions = drop_esm_features(df_sessions)

    colums_age = [x for x in df_sessions.columns.values if x.startswith('f_demographics_age')]
    df_sessions.drop(columns=colums_age, inplace=True)

    colums_gender = [x for x in df_sessions.columns.values if x.startswith('f_demographics_gender')]
    df_sessions.drop(columns=colums_gender, inplace=True)

    colums_wlan_name = [x for x in df_sessions.columns.values if x.startswith('f_internet_connected_WIFI_')]
    df_sessions.drop(columns=colums_wlan_name, inplace=True)

    df_sessions.drop(columns=['f_absentminded_use', 'f_general_use'], inplace=True)

    df_sessions.to_csv(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\labled_data\user-sessions_features_all_labled_more_than_intention_no_personalized.csv')

    with open(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\labled_data\user-sessions_features_all_labled_more_than_intention_normal_age_no_esm_no_personal.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)


def drop_esm_features(df_sessions):
    """
    Drop all ESM feature colums form the dataframe
    :param df_sessions: the dataframe to drop
    :return:
    """
    colums_esm = [x for x in df_sessions.columns.values if x.startswith('f_esm')]
    df_sessions.drop(columns=colums_esm, inplace=True)
    return df_sessions


if __name__ == '__main__':

    # 4. Extract the features from the logs and saves it to the sessions df
    path_list = pathlib.Path(r'D:\usersorted_logs_preprocessed').glob('**/*.pickle')
  #  path_list = pathlib.Path(r'C:\Users\florianb\Downloads').glob('**/*.pickle')

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


    # 5. concat all session and features df from each user to one
    concat_sessions()

    # 6. Create the bag of apps for each sessions (using all session df)
    bag_of_apps_create_vocab()
    bag_of_apps_create_bags()

    # 7. Convert timedeltas to milliseconds and drop unused columns
    drop_sequences()
 #   convert_timedeletas() TODO values are already int

    # 10. On hot encode colums like esm
    # one_hot_encoding_dummies()
    one_hot_encoding_scilearn()

    # 11. Filter outliners
    filter_sessions_outliners_all()

    # 12. Only use users that completed the second questionnaire # TODO weiter vorne in der pipeline machen
    filter_users()

    # 13. reduce feautre dimension by grouping columns together
    reduce_feature_dimension()

    # 13. create labels as targets (only works with onhot encoded data)
    create_labels_single()
    # labeling_combined()

    # 14. If needed - remove personal features like age, gender or absentminded/general use scores
    remove_personalised_features()

    print('preprocessingGroup main() done.')
