import pandas as pd
import scipy.stats as stats

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

path_questionnaire = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\data_MobileRabbitHole.csv'
path_raw = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData'

que_MRH1 = ['SD01', 'SD02_01', 'SD10', 'SD10_09', 'SD14']


def MRH_questionnaire_1():
    df_MRH1_raw = pd.read_csv(f'{path_raw}\MRH1.csv', sep=',')
    df_MRH2 = pd.read_csv(f'{path_raw}\MRH2.csv', sep=',')

    studIDs_mrh2 = df_MRH2['IM01_01']
    df_MRH1 = df_MRH1_raw.loc[df_MRH1_raw['IM01_01'].isin(studIDs_mrh2)]

    mean_age = df_MRH1['SD02_01'].mean()
    mean_gender = df_MRH1['SD01'].mean()
    val_count_age = df_MRH1['SD02_01'].value_counts()
    val_count_gender = df_MRH1['SD01'].value_counts()

    print(mean_age)
    # 26.70967741935484
    # Just completed: 25.4
    print(mean_gender)
    # 1.2580645161290323
    # just completed: 1.2666666666666666
    print(val_count_age)

    print(val_count_gender)
    # 1:    23
    # 2:    8
    # just completed:
    # 1:    11
    # 2:    4

    # plot = df_MRH1['SD02_01'].plot.bar()
    df_MRH1.describe().to_csv(fr'{path_raw}\MRH1_stats.csv')

    df_MRH1.SD02_01.dropna().astype('int64')
    # p = df_MRH1.SD02_01.value_counts(ascending=True).plot(kind = 'barh')
    # p = df_MRH1.SD01.value_counts().sort_index(axis=0).plot(kind = 'bar')

    #
    # selct colums that strat with str
    df_qu_user = df_MRH1[df_MRH1['IM01_01'].values == 'SO23BA'].index.item()
    absentminded = df_MRH1.loc[df_qu_user, df_MRH1.columns.str.startswith('AB01')]
    general = df_MRH1.loc[df_qu_user, df_MRH1.columns.str.startswith('AB02')]

    # print(absentminded.mean())

    # print(absentminded.mean(axis=1))
    # print(general.mean(axis=1))

    # absentminded.hist()
    # plt.show()
    # general.hist()

    # plt.show()

    # k2, p = stats.normaltest(absentminded)
    # k1, p1 = stats.normaltest(general)
    alpha = 0.05  # 1e-3
    # print(p)
    #    [0.67839887 0.47155547 0.90070218 0.73904268 0.39381979 0.22951735
    # 0.1396033  0.10960204 0.55445724 0.25119982 0.40092371 0.15104378
    # 0.75717849]
    # print(p1)
    #  [0.04559977 0.77212013 0.40138207 0.10396851 0.50288916 0.33793807
    # 0.1247074  0.37327643 0.01125694 0.43596354]

    # if(p < alpha):
    # f the p-val is very small, it means it is unlikely that the data came from a normal distribution. For example:
    #     print ("Not normal distribution")


def general_absentminded_use():
    df_MRH1_raw = pd.read_csv(f'{path_raw}\MRH1.csv', sep=',')
    df_MRH2 = pd.read_csv(f'{path_raw}\MRH2.csv', sep=',')
    #  FOR ju05 not mu but ab

    # Select only users that are in both questionnaires
    studIDs_mrh2 = df_MRH2['IM01_01']
    df_MRH1 = df_MRH1_raw.loc[df_MRH1_raw['IM01_01'].isin(studIDs_mrh2)]

    df_MRH1 = df_MRH1.sort_values('IM01_01')
    df_MRH2 = df_MRH2.sort_values('IM01_01')

    df_MRH1.reset_index(drop=True, inplace=True)
    df_MRH2.reset_index(drop=True, inplace=True)

    # print(len(df_MRH1))
    # print(df_MRH1['IM01_01'])
    #
    # print(len(df_MRH2))
    # print(df_MRH2['IM01_01'])

    mrh1_absentminded = df_MRH1.loc[:, df_MRH1.columns.str.startswith('AB01')]
    mrh1_general = df_MRH1.loc[:, df_MRH1.columns.str.startswith('AB02')]
    mean_mrh1_absentmineded = mrh1_absentminded.mean(axis=1)
    mean_mrh1_general = mrh1_general.mean(axis=1)

    # mean_mrh1_absentmineded.hist()
    # plt.show()
    # mean_mrh1_general.hist()
    # plt.show()
    #
    # k2, p = stats.normaltest(mean_mrh1_absentmineded)
    # k1, p1 = stats.normaltest(mean_mrh1_general)
    # print(p)
    # print(p1)

    mrh2_absentminded = df_MRH2.loc[:, df_MRH2.columns.str.startswith('AB01')]
    mrh2_general = df_MRH2.loc[:, df_MRH2.columns.str.startswith('AB02')]
    mean_mrh2_absentmineded = mrh2_absentminded.mean(axis=1)
    mean_mrh2_general = mrh2_general.mean(axis=1)

    stat_absentminded, p_absentminded = stats.ttest_rel(mean_mrh1_absentmineded, mean_mrh2_absentmineded)
    stat_general, p_general = stats.ttest_rel(mean_mrh1_general, mean_mrh2_general)

    print(stat_absentminded, p_absentminded)  # 2.2244055078094105 0.04308068974121597
    print(stat_general, p_general)  # 0.759395726885135 0.4602122087200744

    # df2_cols = df2.columns.drop('stdev')
    # out = [stats.ttest_ind(df2.loc[i, df2_cols], row, equal_var=True, nan_policy='omit')
    #    for i, row in df1.drop(columns=['stdev','Ctrl average']).iterrows()]
    # col = mrh2_general.columns
    # ttest = [stats.ttest_rel(mrh2_general.loc[i, col], row, nan_policy='omit') for i, row in mrh1_general.iterrows()]
    # '[Ttest_relResult(statistic=-0.28734788556634544, pvalue=0.7803523356242351), Ttest_relResult(statistic=0.6882472016116853, pvalue=0.5086464915963445), Ttest_relResult(statistic=-1.6269784336399213, pvalue=0.13818475535118327), Ttest_relResult(statistic=0.0, pvalue=1.0), Ttest_relResult(statistic=-1.9639610121239315, pvalue=0.0811261888458405), Ttest_relResult(statistic=0.4285714285714286, pvalue=0.6783097418055797), Ttest_relResult(statistic=1.5, pvalue=0.16785065605707486), Ttest_relResult(statistic=0.6882472016116853, pvalue=0.5086464915963445), Ttest_relResult(statistic=-3.0736305843324923, pvalue=0.013275877464573012), Ttest_relResult(statistic=-3.279648999660727, pvalue=0.009534613041301043), Ttest_relResult(statistic=1.9639610121239317, pvalue=0.0811261888458405), Ttest_relResult(statistic=-2.7136021011998723, pvalue=0.023856384540122025), Ttest_relResult(statistic=-0.8017837257372731, pvalue=0.44333185016966015), Ttest_relResult(statistic=-1.5, pvalue=0.16785065605707486), Ttest_relResult(statistic=3.086974532565159, pvalue=0.012992786487757368)]

    for i, row in mrh1_general.iterrows():
        col = mrh2_general.columns
        result_general = stats.ttest_rel(mrh2_general.loc[i, col], row, nan_policy='omit')
        df_MRH2.loc[i, 'general_use_ttest_stat'] = result_general[0]
        df_MRH2.loc[i, 'general_use_ttest_p-value'] = result_general[1]

    for i, row in mrh1_absentminded.iterrows():
        col = mrh2_absentminded.columns
        result_absentminded = stats.ttest_rel(mrh2_absentminded.loc[i, col], row, nan_policy='omit')
        df_MRH2.loc[i, 'general_use_ttest_stat'] = result_absentminded[0]
        df_MRH2.loc[i, 'general_use_ttest_p-value'] = result_absentminded[1]

    df_MRH2.to_csv(f'{path_raw}\MRH2_new.csv')


    # As the p-value comes out to be less than 0.05 hence we reject the null hypothesis.
    # So, we have enough proof to claim that the true mean test score is different for cars before and after applying the different engine oil.


def seperate():
    df_qu = pd.read_csv(path_questionnaire, sep=',')
    df_MRH1 = df_qu[df_qu['QUESTNNR'] == 'MRH1']
    df_MRH2 = df_qu[df_qu['QUESTNNR'] == 'MRH2']

    df_MRH1.dropna(axis=1, how='all').to_csv(fr'{path_raw}\MRH1.csv')
    df_MRH2.dropna(axis=1, how='all').to_csv(fr'{path_raw}\MRH2.csv')


if __name__ == '__main__':
    # seperate()
    #MRH_questionnaire_1()

    general_absentminded_use()
