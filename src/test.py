from preprocessingGroup import *


if __name__ == '__main__':
    # # 5. concat all session and features df from each user to one
    # concat_sessions()
    #
    # # 6. Create the bag of apps for each sessions (using all session df)
    # bag_of_apps_create_vocab()
    # bag_of_apps_create_bags()
    #
    # # 7. Convert timedeltas to milliseconds and drop unused columns
    # drop_sequences()
    # convert_timedeletas()
    #
    # # 10. On hot encode colums like esm
    # # one_hot_encoding_dummies()
    # one_hot_encoding_scilearn()
    #
    # # 11. Filter outliners
    # filter_sessions_outliners_all()

    # # 12. Only use users that completed the second questionnaire # TODO weiter vorne in der pipeline machen
    # filter_users()
    #
    # # 13. reduce feautre dimension by grouping columns together
    # reduce_feature_dimension()
    #
    # # 13. create labels as targets (only works with onhot encoded data)
    # create_labels_single()
    # # labeling_combined()
    #
    # # 14. If needed - remove personal features like age, gender or absentminded/general use scores
    # remove_personalised_features()


    df_sessions = pd.read_pickle(
        fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\labled_data\user-sessions_features_all_labled_more_than_intention_normal_age_no_esm_no_personal.pickle')

#    df_sessions['f_session_group_timespan'] = df_sessions['f_session_group_timespan'].apply(lambda x: round(x.total_seconds() * 1000))
    df_sessions['f_session_group_length_active'] = df_sessions['f_session_group_length_active'].apply(lambda x: round(x.total_seconds() * 1000))
    df_sessions['f_session_group_length_active_mean'] = df_sessions['f_session_group_length_active_mean'].apply(lambda x: round(x.total_seconds() * 1000))
    df_sessions['f_session_group_length_active_median'] = df_sessions['f_session_group_length_active_median'].apply(lambda x: round(x.total_seconds() * 1000))
    df_sessions['f_session_group_length_active_sd'] = df_sessions['f_session_group_length_active_sd'].apply(lambda x: (round(x.total_seconds() * 1000)) if x is not pd.Timedelta("nan") else pd.Timedelta("nan"))

    with open(fr'C:\projects\rabbithole\RabbitHoleProcess\data\dataframes\sessiongroups-ml\labled_data\user-sessions_features_all_labled_more_than_intention_normal_age_no_esm_no_personal.pickle', 'wb') as f:
        pickle.dump(df_sessions, f, pickle.HIGHEST_PROTOCOL)

    print('Done.')