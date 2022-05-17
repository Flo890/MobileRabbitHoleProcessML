import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib
matplotlib.rcParams['figure.dpi'] = 200

milkGreen= '#0BCB85'
milkGreenDark= '#267355'
blueish= '#0378C5'
LightBlackishGray= '#6E707C'
DarkBlackishGray= '#6E707C'


# was wollen wir wissen?
# wie wir es ausgelöset, apps websites?
# context zeit ort?
# wie sieht spiralling aus?
# wie lange spiralling session?
# was hat man davor gemacht?

#
dataframe_dir_ml = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\ML'
df_sessions = pd.read_pickle(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle')
df_sessions_a = pd.read_pickle(fr'{dataframe_dir_ml}\analyze-no1hot-withseq-nolabels\user-sessions_features_all.pickle')
df_rh = df_sessions_a[df_sessions_a['f_esm_more_than_intention'] == 'Yes']



def df_analyze_apps():
    # list = [('APP', 'com.instagram.android'), ('APP', 'com.google.android.apps.nexuslauncher'), ('APP', 'com.android.chrome'), ('APP', 'com.google.android.apps.nexuslauncher')]

    # f_sequences_apps [list(['com.whatsapp', 'com.google.android.apps.nexuslauncher', 'com.instagram.android'])]
    list = ['com.whatsapp', 'com.google.android.apps.nexuslauncher', 'com.instagram.android']


    df_new = df_rh.copy()
    df_new['f_app_count_rh'] = 0

    df_new_seq = df_new.dropna(subset=['f_sequences_apps']) #fillna('[]')
    for i, row in df_new_seq.iterrows():
        print('row', i)
        seq = row['f_sequences']
        if seq:
            list = seq[0]
            count = 0
            for item in list:
                if item[0] == 'APP':
                    count += 1
            print('count', count)
            df_new.at[i, 'f_app_count_rh'] = count

    sns.countplot(df_new['f_app_count_rh'], color=milkGreen)
    plt.xlabel('App count in rabbit hole sessions')
    plt.show()

    # f_app_time
    # f_app_count
    #
    # f_app_category_count_
    # f_app_category_time
    # f_clicks f_scrolls

    # df_rh.columns.str.startswith('f_app_time')

def df_analyze_intentions():
    print(df_rh['f_esm_intention'].value_counts())

    counts = df_rh['f_esm_intention'].value_counts()#.sort_values(by=['sessions_count'], ascending=True)
    p = counts.plot(kind='barh', color=milkGreen)
    plt.show()


def esm_per_user():
    grouped_logs = df_sessions.groupby(['studyID'])

    # Iterate over sessions
    for name, df_group in grouped_logs:
        more_no = np.array(df_group[df_group['f_esm_more_than_intention_No'] == 1.0]['f_session_length'])
        more_yes = np.array(df_group[df_group['f_esm_more_than_intention_Yes'] == 1.0]['f_session_length'])
        more_nan = np.array(df_group[df_group['f_esm_more_than_intention_nan'] == 1.0]['f_session_length'])

        type = ["Not more than intention", "More than intention", "nan"]
        count = [more_no.size, more_yes.size, more_nan.size]

        plt.bar(type, count, color=milkGreen)
        plt.title(name)
        plt.show()

        print()
        print(more_no.size, more_yes.size, more_nan.size)

        # print(more_no, more_yes, more_nan)
        #
        # plt.plot(more_no, np.zeros_like(more_no) + 0, 'x', color=milkGreenDark, label="Not more than intention")
        # plt.plot(more_yes, np.zeros_like(more_yes) + 1, 'x', color=milkGreen, label="More than intention")
        # plt.plot(more_nan, np.zeros_like(more_nan) + 2, 'x', color=blueish, label="Nan")
        # plt.legend()
        # plt.title(name)
        # plt.show()


def esm():
    colums_esm = [x for x in df_sessions_a.columns.values if x.startswith('f_esm')]
    print(colums_esm)
    df_esm = df_sessions_a.loc[:, df_sessions_a.columns.str.startswith('f_esm')]
    print(df_esm.columns)

    for esm_item in colums_esm:
        # df_esm = df_sessions_a[esm_item]
        ax = sns.countplot(x=df_sessions_a[esm_item], order=df_sessions_a[esm_item].value_counts(ascending=False).index, color=milkGreen)
        #ax = sns.countplot(x=df['feature_name'], order=df['feature_name'].value_counts(ascending=False).index, color=milkGreen)
        abs_values = df_sessions_a[esm_item].value_counts(ascending=False).values
        plt.bar_label(container=ax.containers[0], labels=abs_values)
        plt.xlabel(esm_item)
        # plt.xticks(rotation=90)
        plt.show()




def test():
    s = df_rh[df_rh['f_session_length'] >= 600000]
    print(df_rh.size) #1214784
    print(s.size) #356976 -> ca 30%


def df_analyze_rh_sessions_time():
    print('analyze rabbithole')
    # df_sessions_a.to_csv(fr'{dataframe_dir_ml}\user-sessions_features_all-analyze.csv')

    session_length_mean = df_rh['f_session_length'].mean() #735253.0153508772 12.2542167min
    print(session_length_mean)

    # context zeit ort?
    # sns.countplot(df_rh['f_hour_of_day'], color = milkGreen)
    # plt.show()
    # sns.countplot(df_rh['f_weekday'], color = milkGreen)
    # plt.show()

    labels_hours = list(range(0, 24))
    weekday_range = list(range(0, 7))
    weekday_labels = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']

    plt.figure()
    plt.subplot(2, 1, 1)
    pw = (df_rh['f_weekday'].groupby(df_rh['f_weekday']).count()).reindex(weekday_range, fill_value=0).plot(kind="bar", ylabel='Counts', xlabel='weekday', color = milkGreen, title=f'Rabbit Hole Sessions')
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)
    pw.set_xticklabels(weekday_labels, rotation=0)

    plt.subplot(2, 1, 2)
    ph = (df_rh['f_hour_of_day'].groupby(df_rh['f_hour_of_day']).count()).reindex(labels_hours, fill_value=0).plot(kind="bar", ylabel='Counts', xlabel='hour of day', color= milkGreen, title=f'Rabbit Hole Sessions')
    ph.set_xticklabels(labels_hours, rotation=0)
    plt.show()



def analyze_esm():
    """
    Plots the count and scatter of session_legnths for esm more than intention per session
    :return:
    """
    print('analyze esm')
    # to_plot1 = np.array(df_sessions[df_sessions['f_esm_more_than_intention_Yes'] == 1.]['f_session_length'])
    # to_plot2 = np.array(df_sessions[df_sessions['target_label'] == 'no_rabbithole']['f_session_length'])

    more_no = np.array(df_sessions[df_sessions['f_esm_more_than_intention_No'] == 1.0]['f_session_length'])
    more_yes = np.array(df_sessions[df_sessions['f_esm_more_than_intention_Yes'] == 1.0]['f_session_length'])
    more_nan = np.array(df_sessions[df_sessions['f_esm_more_than_intention_nan'] == 1.0]['f_session_length'])
    
    type = ["Not more than intention", "More than intention", "nan"]
    count = [more_no.size, more_yes.size, more_nan.size]

    plt.bar(type, count, color=milkGreen)
    plt.show()

    print()
    print(more_no.size, more_yes.size, more_nan.size)

    print(more_no, more_yes, more_nan)

    plt.plot(more_no, np.zeros_like(more_no) + 0, 'x', color=milkGreenDark, label="Not more than intention")
    plt.plot(more_yes, np.zeros_like(more_yes) + 1, 'x', color=milkGreen, label="More than intention")
    plt.plot(more_nan, np.zeros_like(more_nan) + 2, 'x', color=blueish, label="Nan")
    plt.legend()
    plt.show()




if __name__ == '__main__':
    # pd.set_option('display.max_columns', None)
    # analyze_esm()
    # df_analyze_rh_sessions_time()
    # esm_per_user()
    esm()
    # df_analyze_apps()

    # test()
