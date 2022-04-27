import pandas as pd
import pathlib
import pickle
import getFromJson
import prepareTimestamps
import extractSessions
import featureExtraction
import matplotlib.pyplot as plt
import seaborn as sns

dataframe_dir_users = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\users'
dataframe_dir_ml = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\ML'
dataframe_dir_sessions = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\sessions'
dataframe_dir_sessions_features = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\sessions_features'
dataframe_dir_sessions_filtered = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\sessions_filtered'

milkGreen= '#0BCB85'
milkGreenDark= '#267355'
blueish= '#0378C5'
LightBlackishGray= '#6E707C'
DarkBlackishGray= '#6E707C'

def sessions_hist():
    """
    Print session distribution on weekdays and hours of day
    :return:
    """
    print('plot session Hist')
    path_list = pathlib.Path(dataframe_dir_sessions_features).glob('**/*.pickle')

    for data_path in path_list:
        path_in_str = str(data_path)
        print(path_in_str, data_path.stem)
        df = pd.read_pickle(path_in_str)

        labels_hours = list(range(0, 24))
        # "MONDAY": 0,
        #     "TUESDAY": 1,
        #     "WEDNESDAY": 2,
        #     "THURSDAY": 3,
        #     "FRIDAY": 4,
        #     "SATURDAY": 5,
        #     "SUNDAY": 6
        weekday_range = list(range(0, 7))
        weekday_labels = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']

        if not df.empty:
            plt.figure()
            # f_hour_of_day
            # f_weekday
            plt.subplot(2, 1, 1)
            pw = (df['f_weekday'].groupby(df['f_weekday']).count()).reindex(weekday_range, fill_value=0).plot(kind="bar", ylabel='Counts', xlabel='weekday', color = milkGreen, title=f'{data_path.stem}')
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)
            pw.set_xticklabels(weekday_labels, rotation=0)

            plt.subplot(2, 1, 2)
            ph = (df['f_hour_of_day'].groupby(df['f_hour_of_day']).count()).reindex(labels_hours, fill_value=0).plot(kind="bar", ylabel='Counts', xlabel='hour of day', color=milkGreen, title=f'{data_path.stem}')
            ph.set_xticklabels(labels_hours, rotation=0)
            plt.show()

            # plt.subplot(3, 1, 3)
            # #ps = df['f_session_length'].plot(kind="hist", ylabel='Counts', xlabel='hour of day', title=f'{data_path.stem}')
            # sns.boxplot(df['f_session_length'] / pd.Timedelta(seconds=1))
            # #(df['f_session_length'] / pd.Timedelta(seconds=1)).plot(kind='hist', ylabel='Frequency', xlabel='seconds', title=f'{data_path.stem}')
            # plt.show()

            # sns.distplot(df['f_session_length'] / pd.Timedelta(milliseconds=1))
            # plt.show()
            # sns.boxplot(df['f_session_length'] / pd.Timedelta(milliseconds=1))
            # plt.show()


def sessions_hist_all():
    """
    Print session distribution on weekdays and hours of day
    :return:
    """
    print('plot session Hist')
    path = fr'{dataframe_dir_ml}\user-sessions_features_all-analyze.pickle'

    df = pd.read_pickle(path)

    labels_hours = list(range(0, 24))
    # "MONDAY": 0,
    #     "TUESDAY": 1,
    #     "WEDNESDAY": 2,
    #     "THURSDAY": 3,
    #     "FRIDAY": 4,
    #     "SATURDAY": 5,
    #     "SUNDAY": 6
    weekday_range = list(range(0, 7))
    weekday_labels = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']

    if not df.empty:
        plt.figure()
        # f_hour_of_day
        # f_weekday
        plt.subplot(2, 1, 1)
        pw = (df['f_weekday'].groupby(df['f_weekday']).count()).reindex(weekday_range, fill_value=0).plot(kind="bar", ylabel='Counts', xlabel='weekday', color=milkGreen, title='sessions counts')
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)
        pw.set_xticklabels(weekday_labels, rotation=0)

        plt.subplot(2, 1, 2)
        ph = (df['f_hour_of_day'].groupby(df['f_hour_of_day']).count()).reindex(labels_hours, fill_value=0).plot(kind="bar", ylabel='Counts', xlabel='hour of day', color=milkGreen, title='sessions counts')
        ph.set_xticklabels(labels_hours, rotation=0)
        plt.show()

        # plt.subplot(3, 1, 3)
        # #ps = df['f_session_length'].plot(kind="hist", ylabel='Counts', xlabel='hour of day', title=f'{data_path.stem}')
        # sns.boxplot(df['f_session_length'] / pd.Timedelta(seconds=1))
        # #(df['f_session_length'] / pd.Timedelta(seconds=1)).plot(kind='hist', ylabel='Frequency', xlabel='seconds', title=f'{data_path.stem}')
        # plt.show()

        # sns.distplot(df['f_session_length'] / pd.Timedelta(milliseconds=1))
        # plt.show()
        # sns.boxplot(df['f_session_length'] / pd.Timedelta(milliseconds=1))
        # plt.show()


def sessions_count_user():
    """
    plot the session counts for all users
    :return:
    """
    path_list = pathlib.Path(dataframe_dir_sessions_features).glob('**/*.pickle')

    counts = []
    for data_path in path_list:
        path_in_str = str(data_path)
        df = pd.read_pickle(path_in_str)
        counts.append((data_path.stem, len(df)))

    df = pd.DataFrame(counts, columns=['id', 'sessions_count'])
    df.sort_values(by=['sessions_count'], ascending=True, inplace=True)
    p = df.plot(kind='barh', xlabel='counts', color=milkGreen)
    p.set_yticklabels(df['id'].values, rotation=0)

    plt.show()


if __name__ == '__main__':
    sessions_count_user()
    # sessions_hist()
    sessions_hist_all()
