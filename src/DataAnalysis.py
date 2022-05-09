import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pickle

from pandas import ExcelWriter

milkGreen = '#0BCB85'
milkGreenDark = '#267355'
blueish = '#0378C5'
LightBlackishGray = '#6E707C'
DarkBlackishGray = '#6E707C'

dataframe_dir_ml = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\ML'
dataframe_dir_labled = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\ML\labled'


# df_sessions = pd.read_pickle(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle')
# df_sessions_a = pd.read_pickle(fr'{dataframe_dir_ml}\analyze-no1hot-withseq-nolabels\user-sessions_features_all.pickle')


def get_rabbitHoleSessions(df_sessions):
    df = df_sessions[df_sessions['target_label'] == 'rabbit_hole'].reset_index(drop=True)
    return df.copy()


def get_NOrabbitHoleSessions(df_sessions):
    df = df_sessions[df_sessions['target_label'] == 'no_rabbithole'].reset_index(drop=True)
    return df.copy()


def rh_analyze_apps(df_rh):
    print("rh_apps overall")
    # App counts
    # most used app
    # list = [('APP', 'com.instagram.android'), ('APP', 'com.google.android.apps.nexuslauncher'), ('APP', 'com.android.chrome'), ('APP', 'com.google.android.apps.nexuslauncher')]
    # f_sequences_apps [list(['com.whatsapp', 'com.google.android.apps.nexuslauncher', 'com.instagram.android'])]

    df_rh['f_sequences_apps'].fillna('nan', inplace=True)
    df_rh['f_all_app_count'] = 0

    df_apps_count = {}
    df_apps_time = {}

    for i, row in df_rh.iterrows():
        # get total app count per session
        apps = row['f_sequences_apps'][0]
        if apps != 'n':
            df_rh.at[i, 'f_all_app_count'] = len(apps)

        # most used apps per sessions - count

        # most used apps per session - time

    # app count
    sns.countplot(df_rh['f_all_app_count'], color=milkGreen)
    plt.xlabel('App count in rabbit hole sessions')
    # plt.show()

    # most used apps

    # ax = sns.countplot(x=df_sessions_a[esm_item], order=df_sessions_a[esm_item].value_counts(ascending=False).index, color=milkGreen)
    #     #ax = sns.countplot(x=df['feature_name'], order=df['feature_name'].value_counts(ascending=False).index, color=milkGreen)
    #     abs_values = df_sessions_a[esm_item].value_counts(ascending=False).values
    #     plt.bar_label(container=ax.containers[0], labels=abs_values)

    # f_app_category_time
    list_times = ['f_app_time', 'f_app_category_time', ]
    list_counts = ['f_app_count', 'f_app_category_count', 'f_clicks', 'f_scrolls', 'f_scrolls_app_category', 'f_clicks', 'f_clicks_app_category']

    colum_names_count = ['name', 'count']
    column_names_time = ['name', 'time']

    times = []
    counts = []

    for item in list_times:
        times.append(get_counts_all(df_rh, item, column_names_time))

    for item in list_counts:
        counts.append(get_counts_all(df_rh, item, colum_names_count))

    # with ExcelWriter(fr'{dataframe_dir_ml}\analyze_counts_times\counts.xls', if_sheet_exists="overlay",) as writer:
    #     for n, df in enumerate(counts):
    #         df.to_excel(writer,'sheet%s' % n)
    # with ExcelWriter(fr'{dataframe_dir_ml}\analyze_counts_times\times.xls', if_sheet_exists="overlay",) as writer:
    #     for n, df in enumerate(times):
    #         df.to_excel(writer,'sheet%s' % n)


def get_counts_all(df_rh,  column_prefix, colum_names):
    df = df_rh.loc[:, df_rh.columns.str.startswith(column_prefix)]
    df_all = {}
    for colum in df.columns:
        sum_count = df[colum].values.sum()
        df_all[colum] = sum_count

    df_all = pd.DataFrame.from_dict(df_all, orient='index').reset_index(drop=False)
    df_all.columns = colum_names
    df_all = df_all.sort_values(by=colum_names[1], ascending=False).reset_index(drop=True)
    print(df_all)
    df_all.to_csv(fr'{dataframe_dir_ml}\analyze_counts_times\analyze_{column_prefix}.csv')
    return df_all


def rh_analyze_sessionlenghts(df_rh):
    print("rh_sessionlengths")
    df_length = df_rh['f_session_length']
    session_length_mean = df_length.mean()  # 735253.0153508772 12.2542167min
    print(session_length_mean)

    sns.boxplot(df_length)
    # TODO Barplots with Bins
    plt.show()

    bins = [0, 60000, 300000, 600000, 1200000, 1800000, 2400000, 3000000, 3600000, 5400000, 7200000, 9000000, 10800000]
    # labels = ['0', '1min', '5min', '10min', '20min', '30min', '40min', '50min', '60min', '90min', '120min', '150min']
    labels = [0, 1, 5, 10, 20, 30, 40, 50, 60, 90, 120, 150, 180]
    counts, edges, bars = plt.hist(df_length.values, bins=bins, edgecolor="k")
    # n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
    plt.xlabel('milliseconds')
    plt.xticks(bins, rotation='vertical')
    plt.bar_label(bars)

    plt.show()


def rh_analyze_context(df_rh):
    print("analyze rh_context")
    # df_sessions_a.to_csv(fr'{dataframe_dir_ml}\user-sessions_features_all-analyze.csv')

    labels_hours = list(range(0, 24))
    weekday_range = list(range(0, 7))
    weekday_labels = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']

    plt.figure()
    plt.subplot(2, 1, 1)
    # df_rh.columns.str.startswith('f_weekday')
    df_wd = df_rh.loc[:, df_rh.columns.str.startswith('f_weekday')]
    df_weekdays = pd.get_dummies(df_wd).idxmax(1)
    print(df_weekdays)
    # print(df_weekdays.groupby(level=0))
    df_weekdays.replace({'f_weekday_': ''}, regex=True, inplace=True)
    df_weekdays.astype('float')
    print(df_weekdays)

    weekdays = {}
    # TODO create sessions labled dataframe without onehotenhoding??
    # pw = (df_weekdays.value_counts.plot(kind="bar", ylabel='Counts', xlabel='weekday', color=milkGreen, title=f'Rabbit Hole Sessions')
    pw = (df_weekdays.groupby(level=0).count()).reindex(weekday_range, fill_value=0).plot(kind="bar", ylabel='Counts', xlabel='weekday', color=milkGreen, title=f'Rabbit Hole Sessions')
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)
    pw.set_xticklabels(weekday_labels, rotation=0)

    # plt.subplot(2, 1, 2)
    # ph = (df_rh['f_hour_of_day'].groupby(df_rh['f_hour_of_day']).count()).reindex(labels_hours, fill_value=0).plot(kind="bar", ylabel='Counts', xlabel='hour of day', color=milkGreen,
    #                                                                                                               title=f'Rabbit Hole Sessions')
    # ph.set_xticklabels(labels_hours, rotation=0)
    plt.show()


def rh_analyze_intentions(df_rh):
    print(df_rh['f_esm_intention'].value_counts())

    counts = df_rh['f_esm_intention'].value_counts()  # .sort_values(by=['sessions_count'], ascending=True)
    counts.plot(kind='barh', color=milkGreen)
    plt.show()


def rh_analyze_app_sequences(df_rh):



if __name__ == '__main__':
    path = [rf'{dataframe_dir_labled}\user-sessions_features_labeled_more_than_intention.pickle',
            rf'{dataframe_dir_labled}\user-sessions_features_labeled_f_esm_more_than_intention_Yes_f_esm_agency_0.0.pickle',
            rf'{dataframe_dir_labled}\user-sessions_features_labeled_f_esm_more_than_intention_Yes_f_esm_regret_7.0.pickle',
            rf'{dataframe_dir_labled}\user-sessions_features_labeled_f_esm_more_than_intention_Yes_f_esm_track_of_time_6.0.pickle']

    df_sessions = pd.read_pickle(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle')

    df_sessions_labled = pd.read_pickle(path[0])

    # print(df_sessions.size)
    # print(df_sessions_labled.size)

    df_rabbitHole = get_rabbitHoleSessions(df_sessions_labled)
    df_no_rabbitHole = get_NOrabbitHoleSessions(df_sessions_labled)

    # df_rabbitHole.to_csv()

    # rh_analyze_intentions(df_rabbitHole)
    # rh_analyze_context(df_rabbitHole)
    # rh_analyze_sessionlenghts(df_rabbitHole)
    rh_analyze_apps(df_sessions)
