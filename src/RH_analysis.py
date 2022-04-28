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
    list = [('APP', 'com.instagram.android'), ('APP', 'com.google.android.apps.nexuslauncher'), ('APP', 'com.android.chrome'), ('APP', 'com.google.android.apps.nexuslauncher')]

    # for item in list:


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
    pd.set_option('display.max_columns', None)
    # analyze_esm()
    # df_analyze_rh_sessions_time()
    esm_per_user()
