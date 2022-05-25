import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pickle
import ML_helpers
import statistics
import matplotlib


matplotlib.rcParams["figure.dpi"] = 250
size=12
ticksize = 12
legendsize=13
plt.rc('font', size=size) #controls default text size
plt.rc('axes', titlesize=size) #fontsize of the title
plt.rc('axes', labelsize=size) #fontsize of the x and y labels
plt.rc('xtick', labelsize=ticksize) #fontsize of the x tick labels
plt.rc('ytick', labelsize=ticksize) #fontsize of the y tick labels
plt.rc('legend', fontsize=legendsize) #fontsize of the legend

#plt.rcParams['text.usetex'] = True
# preamb = r"\usepackage{bm} \usepackage{mathtools}"
# params = {'text.latex.preamble' : preamb }
# plt.rcParams.update(params)
#plt.rcParams['font.family'] = "computer modern roman"




milkGreen = '#0BCB85'
milkGreenDark = '#267355'
blueish = '#0378C5'
LightBlackishGray = '#6E707C'
DarkBlackishGray = '#6E707C'

dataframe_dir_ml = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\ML'
dataframe_dir_results = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\results'
dataframe_dir_labled = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\ML\labled_data\labled'
dataframe_dir_ml_labeled = f'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\ML\labled_data'



def get_rabbitHoleSessions(df_sessions):
    """
    Get all sessions labled as a rabbit holse
    :param df_sessions: the labled dataset of al sessions
    :return: rabbit hole sessions df
    """
    df = df_sessions[df_sessions['target_label'] == 'rabbit_hole'].reset_index(drop=True)
    return df.copy()


def get_NotRabbitHoleSessions(df_sessions):
    """
    Get all sessions labled as a not rabbit holse
    :param df_sessions: the labled dataset of all sessions
    :return: non rabbit hole sessions df
    """
    df = df_sessions[df_sessions['target_label'] == 'no_rabbithole'].reset_index(drop=True)
    return df.copy()


def rh_relative_frequency(df_rh):
    """
    Calculate realtive frequency for differnt features per sessions
    :param df_rh: dataframe of sessions
    """
    print("relative freuquency")
    df_rh['f_sequences_apps'].fillna('nan', inplace=True)
    df_rh['f_all_app_count'] = 0
    df_rh['f_all_app_category_count'] = 0

    for i, row in df_rh.iterrows():
        # f_app_time_columns = [x for x in df_rh.columns.values if x.startswith('f_app_time')]

        apps = row['f_sequences_apps'][0]

        if apps != 'n':
            # Expand the app sequence with time spent on app and app category
            # expand_app_seq(apps, row)
            df_rh.at[i, 'f_all_app_count'] = len(apps)

        df_app_category_count = df_rh.loc[i, df_rh.columns.str.startswith('f_app_category_count')]
        f_app_category_sum = df_app_category_count.values.sum()
        df_rh.at[i, 'f_all_app_category_count'] = f_app_category_sum

        f_app_count = df_rh.loc[i, df_rh.columns.str.startswith('f_app_count')]
        column_app_count = f_app_count[f_app_count > 0].index.values
        for app in column_app_count:
            if f'{app}_relative_frequency' not in df_rh.columns:
                df_rh[f'{app}_relative_frequency'] = 0

            df_rh.at[i, f'{app}_relative_frequency'] = f_app_count[app] / len(apps)

        f_app_time = df_rh.loc[i, df_rh.columns.str.startswith('f_app_time')]
        column_app_time = f_app_time[f_app_time > 0].index.values
        for app in column_app_time:
            if f'{app}_relative_frequency' not in df_rh.columns:
                df_rh[f'{app}_relative_frequency'] = 0

            df_rh.at[i, f'{app}_relative_frequency'] = f_app_time[app] / row['f_session_length']

        f_app_category_count = df_rh.loc[i, df_rh.columns.str.startswith('f_app_category_count')]
        column_app_category_count = f_app_category_count[f_app_category_count > 0].index.values
        for app in column_app_category_count:
            if f'{app}_relative_frequency' not in df_rh.columns:
                df_rh[f'{app}_relative_frequency'] = 0

            df_rh.at[i, f'{app}_relative_frequency'] = f_app_category_count[app] / f_app_category_sum

        f_app_category_time = df_rh.loc[i, df_rh.columns.str.startswith('f_app_category_time')]
        column_app_category_time = f_app_category_time[f_app_category_time > 0].index.values
        for app in column_app_category_time:
            if f'{app}_relative_frequency' not in df_rh.columns:
                df_rh[f'{app}_relative_frequency'] = 0

            df_rh.at[i, f'{app}_relative_frequency'] = f_app_category_time[app] / row['f_session_length']

    df_rh.to_csv(rf'{dataframe_dir_ml_labeled}\features_labeled_more_than_intention_rabbitHoleSessions_analyzed.csv')
    # with open(rf'{dataframe_dir_labled}\features_labeled_more_than_intention_rabbitHoleSessions_analyzed.pickle', 'wb') as f:
    #    pickle.dump(df_rh, f, pickle.HIGHEST_PROTOCOL)



def rh_analyze_apps(df_rh):
    print("rh_apps overall")
    # App counts
    # most used app
    # list = [('APP', 'com.instagram.android'), ('APP', 'com.google.android.apps.nexuslauncher'), ('APP', 'com.android.chrome'), ('APP', 'com.google.android.apps.nexuslauncher')]
    # f_sequences_apps [list(['com.whatsapp', 'com.google.android.apps.nexuslauncher', 'com.instagram.android'])]

    df_rh['f_sequences_apps'].fillna('nan', inplace=True)
    df_rh['f_all_app_count'] = 0

    df_apps_count_start = {}
    df_apps_count_end = {}

    # TODO apps time, categories?
    df_apps_time = {}

    for i, row in df_rh.iterrows():
        # get total app count per session
        apps = row['f_sequences_apps'][0]
        if apps != 'n':
            # Expand the app sequence with time spent on app and app category
            # expand_app_seq(apps, i, row)
            app_counts = len(apps)

            if apps[0] not in df_apps_count_start:
                df_apps_count_start[apps[0]] = 1
            else:
                df_apps_count_start[apps[0]] += 1

            if len(apps) > 1:
                if apps[-1] not in df_apps_count_end:
                    df_apps_count_end[apps[-1]] = 1
                else:
                    df_apps_count_end[apps[-1]] += 1

            df_rh.at[i, 'f_all_app_count'] = app_counts

        # most used apps per sessions - count
        # most used apps per session - time

    colum_names_count = ['name', 'count']

    # app count
    df_rh = df_rh.drop(df_rh.index[df_rh['f_all_app_count'] == 0]).drop(df_rh.index[df_rh['f_all_app_count'] > 100])
    pl = sns.countplot(df_rh['f_all_app_count'], color=milkGreen)
    plt.xlabel('App count')
    plt.show()

    app_count = df_rh['f_all_app_count'].describe()
    print(app_count)
    # count    447.000000
    # mean       4.955257
    # std        9.584309
    # min        0.000000
    # 25%        1.000000
    # 50%        3.000000
    # 75%        5.000000
    # max      109.000000
    # Name: f_all_app_count,

    #RH
    #     count    385.000000
    # mean       5.189610
    # std        6.852455
    # min        1.000000
    # 25%        2.000000
    # 50%        3.000000
    # 75%        5.000000
    # max       78.000000
    # Name: f_all_app_count, dtype: float64

    # no RH
    #     count    13486.000000
    # mean         5.526324
    # std          7.931362
    # min          1.000000
    # 25%          2.000000
    # 50%          3.000000
    # 75%          6.000000
    # max         97.000000
    # Name: f_all_app_count, dtype: float64

    # all
    #     count    13871.000000
    # mean         5.516978
    # std          7.903381
    # min          1.000000
    # 25%          2.000000
    # 50%          3.000000
    # 75%          6.000000
    # max         97.000000
    # Name: f_all_app_count, dtype: float64

    # df_count_start = pd.DataFrame.from_dict(df_apps_count_start, orient="index").reset_index(drop=False)
    # df_count_start.columns = colum_names_count
    # df_count_start = df_count_start.sort_values(by=colum_names_count[1], ascending=False).reset_index(drop=True)
    # print(df_count_start)
    # p = df_count_start.plot(kind="bar", ylabel='Counts', xlabel='count of starter apps', color=milkGreen, title=f'Rabbit Hole Sessions')
    # p.set_xticklabels(df_count_start['name'], rotation=90)
    # plt.show()
    #
    # df_count_end = pd.DataFrame.from_dict(df_apps_count_end, orient="index").reset_index(drop=False)
    # df_count_end.columns = colum_names_count
    # df_count_end = df_count_end.sort_values(by=colum_names_count[1], ascending=False).reset_index(drop=True)
    # print(df_count_end)
    # p = df_count_end.plot(kind="bar", ylabel='Counts', xlabel='count of end apps', color=milkGreen, title=f'Rabbit Hole Sessions')
    # p.set_xticklabels(df_count_end['name'], rotation=90)
    # plt.show()

    # most used apps

    # ax = sns.countplot(x=df_sessions_a[esm_item], order=df_sessions_a[esm_item].value_counts(ascending=False).index, color=milkGreen)
    #     #ax = sns.countplot(x=df['feature_name'], order=df['feature_name'].value_counts(ascending=False).index, color=milkGreen)
    #     abs_values = df_sessions_a[esm_item].value_counts(ascending=False).values
    #     plt.bar_label(container=ax.containers[0], labels=abs_values)

    # f_app_category_time
    list_times = ['f_app_time', 'f_app_category_time', ]
    list_counts = ['f_app_category_count', 'f_app_count', 'f_scrolls_app_category', 'f_clicks_app_category']
    list_counts_sort = ['f_clicks_', 'f_scrolls_']

    colum_names_count = ['name', 'count']
    column_names_time = ['name', 'time in min']

    times = []
    counts = []

    for item in list_times:
        times.append(get_counts_all(df_rh, item, column_names_time, time=True))
        get_counts_average(df_rh, item, time=True)

    for item in list_counts:
        counts.append(get_counts_all(df_rh, item, colum_names_count))
        get_counts_average(df_rh, item)

    for item in list_counts_sort:
        counts.append(get_counts_all(df_rh, item, colum_names_count, drop=True))
        get_counts_average(df_rh, item)


def get_counts_all(df_rh, column_prefix, colum_names, drop=False, time=False):
    df = df_rh.loc[:, df_rh.columns.str.startswith(column_prefix)]
    if drop:
        to_drop = [x for x in df.columns.values if x.startswith(f'{column_prefix}app_category')]
        df = df.drop(columns=to_drop)
    df_all = {}
    for colum in df.columns:
        sum_count = df[colum].values.sum()
        df_all[colum] = sum_count

    df_all = pd.DataFrame.from_dict(df_all, orient='index').reset_index(drop=False)
    df_all.columns = colum_names
    df_all = df_all.sort_values(by=colum_names[1], ascending=False).reset_index(drop=True)
    df_all = df_all.set_index('name')
    # for time
    if time:
        df_all = df_all.apply(lambda x: round(((x / 1000) / 60), 2))

    print(df_all.head(5))
    counts = df_all.head(60)
    abs_values = counts[colum_names[1]].values
    p = counts.plot(kind='barh', color=milkGreen, title=column_prefix)
    plt.bar_label(container=p.containers[0], labels=abs_values)
    plt.show()

    # print(df_all)
    # df_all.to_csv(fr'{dataframe_dir_ml}\analyze_counts_times\analyze_{column_prefix}.csv')
    return df_all


def get_counts_average(df_rh, column_prefix, time=False):
    df = df_rh.loc[:, df_rh.columns.str.startswith(column_prefix)].describe()
    df = df.transpose()
    print(df)
    df = df.sort_values('mean', ascending=False)
    # df.sort_values(by='mean', axis=1, ascending=False)
    df.reset_index(inplace=True, drop=False)
    if time:
        df_mean = df['mean'].apply(lambda x: (x / 1000)).head(30)
    else:
        df_mean = df.loc[:, 'mean'].head(30)

    plt.rcdefaults()
    fig, ax = plt.subplots()

    ax.barh(df.loc[:, 'index'].head(30), df_mean, color=milkGreen)
    if time:
        ax.set_xlabel('Seconds')
    else:
        ax.set_xlabel('Count')
    ax.set_title(f'Mean {column_prefix}')
    if time:
        values = df.loc[:, 'mean'].apply(lambda x: round((x / 1000), 2)).head(30).values
    else:
        values = df.loc[:, 'mean'].apply(lambda x: round(x, 2)).head(30).values
    plt.bar_label(container=ax.containers[0], labels=values, label_type='edge')
    # fig.set_dpi(150)
    plt.show()

    # fig, ax = plt.subplots()
    # ax = plt.barh(df.loc[:, 'index'].head(30), df_mean, color=milkGreen)
    # ax.title = f'Mean {column_prefix}'
    # ax.label = f'Mean {column_prefix}'
    # ax.xlabel = f'Mean {column_prefix}'
    # ax.set_xlabel('mean')
    # # ax.set_xlabel(f'Mean {column_prefix}')
    # ax.set_title('title')
    # # abs_values =  df.loc[:, 'mean'].head(30)
    # if time:
    #     values = df.loc[:, 'mean'].apply(lambda x: round(((x / 1000) / 60), 2)).head(30).values
    # else:
    #     values = df.loc['mean'].head(30).values
    # plt.bar_label(container=pj, labels=values, label_type='center')
    # plt.show()

    df.to_csv(fr'{dataframe_dir_results}\descriptives\norh_descriptives_stats_{column_prefix}.csv')


def expand_app_seq(app_seq, row):
    """
    Expand the app sequence list of a session with the time spent of each app and the app category
    :param app_seq:
    :return: the expanded sequence
    """
    # TODO what happens with reduced apps? time is mapped to other? dont list app in seq, create extra other?
    seq_new = []
    for app in app_seq:
        print(app)
        f_app_time = f'f_app_time_{app}'
        time = row[f_app_time]

        category = ML_helpers.get_app_category(app)
        # f_app_category_time = f'f_app_category_time_{category}'
        # f_app_category_count = f'f_app_category_count_{category}'

        # if not time.empty:
        tup = (app, category)
        seq_new.append(tup)
        # else:
    return seq_new


def analyze_esm_features(df_rh):
    print('analyze esm features rh')
    likertScale_title = 'Likert Scale Levels (0: Strongly disagree, 7: Strongly agree)'
    answer_title = 'ES answer'

    # Check finished - f_esm_finished_intention_nan - Did you <b>finish</b> your intention?
    plot_esm_count(df_rh, 'f_esm_finished_intention_', 'Did you finish your intention?', answer_title, 'string')

    # Check emotion - f_esm_emotion_surprise-astonishment - Which pair of <b>emotion</b> corresponds best to how you felt this usage session?
    plot_esm_count(df_rh, 'f_esm_emotion_', 'Which pair of emotion corresponds best to how you felt this usage session?', answer_title, 'string')

    # Check regret -  f_esm_regret_nan -  >I feel <b>regret</b> for part of my phone use
    plot_esm_count(df_rh, 'f_esm_regret_', 'I feel regret for part of my phone use.', likertScale_title, 'float')

    # CHeck track of time - f_esm_track_of_time_nan-  >I have <b>lost track of time</b> while using my phone
    plot_esm_count(df_rh, 'f_esm_track_of_time_', 'I have lost track of time while using my phone.', likertScale_title, 'float')

    # Check track of space - f_esm_track_of_space_nan - I had a <b>good sense of my surroundings</b> while using my phone
    plot_esm_count(df_rh, 'f_esm_track_of_space_', 'I had a good sense of my surroundings while using my phone.', likertScale_title, 'float')

    # Check agency - f_esm_agency_nan -  I had a strong sense of <b>agency<
    plot_esm_count(df_rh, 'f_esm_agency_', 'I had a strong sense of agency.', likertScale_title, 'float')


def plot_esm_count(df_rh, feature_name_prefix, title, x_lable, type_of_item):
    df_to_show = onhotencoding_anti(df_rh, feature_name_prefix, type_of_item)
    print(df_to_show.value_counts().sort_index(axis=0))
    fig = plt.figure(figsize=(16, 9), dpi=250)
    pw = (df_to_show.value_counts().sort_index(axis=0)).plot(kind="bar", ylabel='Counts', xlabel=x_lable, title=title, color=milkGreen)
    abs_values = df_to_show.value_counts().sort_index(axis=0).values
    plt.bar_label(container=pw.containers[0], labels=abs_values, padding=-2)
    fig.tight_layout()
    plt.show()


def onhotencoding_anti(df_rh, prefix, type):
    df_wd = df_rh.loc[:, df_rh.columns.str.startswith(prefix)]
    df_return = pd.get_dummies(df_wd).idxmax(1)
    df_return.replace({prefix: ''}, regex=True, inplace=True)
    df_return.astype(type)
    # print(df_return)
    return df_return


def rh_analyze_sessionlenghts(df_rh):
    print("rh_sessionlengths")
    session_length_mean = df_rh['f_session_length'].describe()  # 735253.0153508772 12.2542167min
    print(session_length_mean)

    session_length_mean.to_csv(fr'{dataframe_dir_results}\sessions_lengths_stats.csv')

    df_length = df_rh['f_session_length'].apply(lambda x: (x / 1000) / 60)

    #fig = plt.figure()
    sns.boxplot(df_length, color=milkGreen)
    plt.title("Sessions lengths distribution")
    plt.xlabel("minutes")
    #fig.set_dpi(300)
    plt.show()

    #fig = plt.figure()
    # bins = [0, 60000, 300000, 600000, 1200000, 1800000, 2400000, 3000000, 3600000, 5400000, 7200000, 9000000, 10800000]
    bins = [0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 90, 120, 150]
    labels = ['0', '1min', '5min', '10min', '20min', '30min', '40min', '50min', '60min', '90min', '120min', '150min']
    # labels = [0, 1, 5, 10, 20, 30, 40, 50, 60, 90, 120, 150, 180]
    counts, edges, bars = plt.hist(df_length.values, bins=bins, edgecolor=milkGreenDark, color=milkGreen)
    # n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)

    plt.xlabel('minutes')
    plt.ylabel('count')
    plt.title("Sessions lengths in rabbit hole sessions")
    plt.xticks(bins, rotation='vertical')
    plt.bar_label(bars)
    #fig.set_dpi(200)
    plt.show()


def analyze_per_user(df_rh):
    print("analyzes per user")
    df_rh['f_sequences_apps'].fillna('nan', inplace=True)

    # different apps:
    grouped_logs = df_rh.groupby(['studyID'])
    colums_weekday = [x for x in df_rh.columns.values if x.startswith('f_weekday_')]
    f_count_app = [x for x in df_rh.columns.values if x.startswith('f_app_count_')]


    count_list_perday_mean = []
    sessionLength_mean = []

    # Iterate over sessions
    for name, df_group in grouped_logs:
        count_list_perday_peruser = []
        mean_Sessions_length = []

        for colum in colums_weekday:
            # all sessions on weekday
            # session counts per day
            weekday = df_group[df_group[colum] == 1]
            day_sessions_count = len(weekday)
            count_list_perday_peruser.append(day_sessions_count)

            # mean session length per day
            sessionLength = weekday['f_session_length'].mean()
            mean_Sessions_length.append(sessionLength)

            # app count
            # apps = df_group['f_sequences_apps'].apply(lambda x: len(x[0]))
            # if apps != 'n':

            # f_app_count_com.google.android.gm

        count_list_perday_mean.append(statistics.mean(count_list_perday_peruser))
        sessionLength_mean.append(statistics.mean(mean_Sessions_length))

    print("Session Length mean:", statistics.mean(sessionLength_mean))
    print("session length std:", statistics.stdev(sessionLength_mean))

    #     # all
    #     Session Length mean: 82.625
    # session length std: 49.5001446757145

    # mean session counts per day
    print("session count mean of mean", statistics.mean(count_list_perday_mean))
    print("session count std:", statistics.stdev(count_list_perday_mean))
    # rabbit hole sessions:
    # session count mean of mean 2.4293478260869565
    # session count std: 2.399936696299521

    # no rh sessions:
    # session count mean of mean 80.39
    # session count std: 48.328874435131084

    # all:
    # session count mean of mean 82.625
    # session count std: 49.5001446757145

    # mean app count per day


    # On an average day, participants used their phone at 23:41 for the last time and at 7:55 for the first time
    # the participants used a total number of 4,857 different apps, with each participant having used 13.10 (SD = 5.93) distinct apps per day, on average.
    # Whereas the most popular app - WhatsApp was on average used 40.62 (SD = 40.32) times per day, per participant,
    # unlocked their smartphones an average number of 47.73 (SD = 30.86) times per day. Furthermore,


def rh_analyze_context(df_rh):
    print("analyze rh_context")
    # df_sessions_a.to_csv(fr'{dataframe_dir_ml}\user-sessions_features_all-analyze.csv')

    labels_hours = list(range(0, 24))
    weekday_range = list(range(0, 7))
    weekday_labels = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
    # weekday_labels = ['Thur', 'Fri', 'Sat', 'Sun']

    df_hours = onhotencoding_anti(df_rh, prefix='f_hour_of_day_', type='float')
    df_H = df_hours.value_counts().reset_index(drop=False)
    df_H['index'] = df_H['index'].astype(float)
    # print(df_hours)
    df_weekday = onhotencoding_anti(df_rh, prefix='f_weekday_', type='float')
    # print(df_weekday)

    plt.figure()
    plt.subplot(2, 1, 1)
    ps = (df_weekday.value_counts().sort_index(axis=0)).plot(kind="bar", ylabel='Counts', xlabel='Weekday', color=milkGreen)
    abs_values = df_weekday.value_counts().sort_index(axis=0).values
    plt.bar_label(container=ps.containers[0], labels=abs_values)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)
    ps.set_xticklabels(weekday_labels, rotation=0)

    plt.subplot(2, 1, 2)
    pw = (df_H.sort_values('index', axis=0, ascending=True).reset_index(drop=True).set_index('index', drop=True)).plot(kind="bar", ylabel='Counts', xlabel='hour of day', color=milkGreen)
    abs_values = df_H.sort_values('index', axis=0, ascending=True).reset_index(drop=True).set_index('index', drop=True)[0].values
    plt.bar_label(container=pw.containers[0], labels=abs_values)
    # pw.set_xticklabels(labels_hours, rotation=0)

    plt.show()


def rh_analyze_intentions(df_rh):
    print(df_rh['f_esm_intention'].value_counts())

    counts_all = df_rh['f_esm_intention'].value_counts()  # .sort_values(by=['sessions_count'], ascending=True)
    counts = counts_all.head(30)
    abs_values = counts.values
    #fig = plt.figure()
    p = counts.plot(kind='barh', color=milkGreen, title="Intention counts")
    plt.bar_label(container=p.containers[0], labels=abs_values)
    #fig.set_dpi(150)
    plt.show()

    counts_all.to_csv(fr'{dataframe_dir_results}\intentions_counts_norh')

def rh_analyze_demographics_bins(df_rh):
    print("analyze demographivs")
    df_age = onhotencoding_anti(df_rh, prefix='f_demographics_age_', type='float')
    df_gender = onhotencoding_anti(df_rh, prefix='f_demographics_gender_', type='float')

    gender_range = list(range(0, 2))
    age_range = list(range(0, 5))
    gender_lables = ['female', 'male']
    age_lables = ['<19', '20-29', '30-39', '40-49', '50-59']  # , '60-69', '<70']

    # print((df_gender.groupby(level=0).count()).reindex(gender_range, fill_value=0))
    # print((df_age.groupby(level=0).count()).reindex(gender_range, fill_value=0))

    print(df_gender.value_counts().sort_index(axis=0))  # .reindex(gender_range, fill_value=0).sort_index(axis=0))
    print(df_age.value_counts().sort_index(axis=0))

    plt.figure()
    plt.subplot(2, 1, 1)
    ps = (df_gender.value_counts().sort_index(axis=0)).plot(kind="bar", ylabel='Counts', xlabel='Gender', color=milkGreen, title=f'Rabbit Hole Sessions')
    abs_values_gender = df_gender.value_counts().sort_index(axis=0).values
    plt.bar_label(container=ps.containers[0], labels=abs_values_gender)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)
    ps.set_xticklabels(gender_lables, rotation=0)

    plt.subplot(2, 1, 2)
    pw = (df_age.value_counts().sort_index(axis=0)).plot(kind="bar", ylabel='Counts', xlabel='Age', color=milkGreen)
    abs_values_age = df_age.value_counts().sort_index(axis=0).values
    plt.bar_label(container=pw.containers[0], labels=abs_values_age)
    pw.set_xticklabels(age_lables, rotation=0)

    plt.show()


if __name__ == '__main__':
    paths = [rf'{dataframe_dir_labled}\user-sessions_features_labeled_more_than_intention.pickle',
             rf'{dataframe_dir_labled}\user-sessions_features_labeled_f_esm_more_than_intention_Yes_f_esm_agency_0.0.pickle',
             rf'{dataframe_dir_labled}\user-sessions_features_labeled_f_esm_more_than_intention_Yes_f_esm_regret_7.0.pickle',
             rf'{dataframe_dir_labled}\user-sessions_features_labeled_f_esm_more_than_intention_Yes_f_esm_track_of_time_6.0.pickle']

    # path = fr'{dataframe_dir_ml_labeled}\user-sessions_features_all_labled_more_than_intention.pickle'
    # path = fr'{dataframe_dir_ml_labeled}\user-sessions_features_labeled_more_than_intention_with_esm.pickle'
    path = fr'{dataframe_dir_ml_labeled}\user-sessions_features_all_labled_more_than_intention_normal_age_with_esm.pickle'

    # df_sessions = pd.read_pickle(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle')

    df_sessions_labled = pd.read_pickle(path)

    # print(df_sessions.size)
    # print(df_sessions_labled.size)

    df_rabbitHole = get_rabbitHoleSessions(df_sessions_labled)
    df_no_rabbitHole = get_NotRabbitHoleSessions(df_sessions_labled)

    # df_rabbitHole.to_csv()

    # rh_analyze_intentions(df_sessions_labled)

    # rh_analyze_context(df_rabbitHole)
    rh_analyze_sessionlenghts(df_sessions_labled)

    # rh_analyze_apps(df_no_rabbitHole)

    # analyze_per_user(df_rabbitHole)

    # rh_analyze_demographics(df_rabbitHole)

    # analyze_esm_features(df_rabbitHole)

    # rh_realtive_frequency(df_rabbitHole)
    #  plot_relative_frequency()
