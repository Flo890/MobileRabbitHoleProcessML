import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

path_questionnaire = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\data_MobileRabbitHole.csv'
path_raw = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData'

que_MRH1 = ['SD01', 'SD02_01', 'SD10', 'SD10_09', 'SD14']

def MRH_questionnaire_1():
    df_MRH1 = pd.read_csv(f'{path_raw}\MRH1.csv',  sep=',')

    mean_age = df_MRH1['SD02_01'].mean()
    mean_gender = df_MRH1['SD01'].mean()

    print(mean_age)
    print(mean_gender)

    # plot = df_MRH1['SD02_01'].plot.bar()
    df_MRH1.describe().to_csv(fr'{path_raw}\MRH1_stats.csv')

    df_MRH1.SD02_01.dropna().astype('int64')
    # p = df_MRH1.SD02_01.value_counts(ascending=True).plot(kind = 'barh')
    # p = df_MRH1.SD01.value_counts().sort_index(axis=0).plot(kind = 'bar')

    #
    #selct colums that strat with str
    absentminded = df_MRH1.loc[:, df_MRH1.columns.str.startswith('AB01')]

    general = df_MRH1.loc[:, df_MRH1.columns.str.startswith('AB02')]

    print(absentminded.mean(axis=1))
    print(general.mean(axis=1))

    absentminded.hist()
    #plt.show()
    general.hist()

    #plt.show()

    k2, p = stats.normaltest(absentminded)
    k1, p1 = stats.normaltest(general)
    alpha = 0.05 # 1e-3
    print(p)
    #    [0.67839887 0.47155547 0.90070218 0.73904268 0.39381979 0.22951735
    # 0.1396033  0.10960204 0.55445724 0.25119982 0.40092371 0.15104378
    # 0.75717849]
    print(p1)
     #  [0.04559977 0.77212013 0.40138207 0.10396851 0.50288916 0.33793807
     # 0.1247074  0.37327643 0.01125694 0.43596354]

    # if(p < alpha):
    # f the p-val is very small, it means it is unlikely that the data came from a normal distribution. For example:
    #     print ("Not normal distribution")




def seperate():
    df_qu = pd.read_csv(path_questionnaire,  sep=',')
    df_MRH1 = df_qu[df_qu['QUESTNNR'] == 'MRH1']
    df_MRH2 = df_qu[df_qu['QUESTNNR'] == 'MRH2']

    df_MRH1.dropna(axis=1, how='all').to_csv(fr'{path_raw}\MRH1.csv')
    df_MRH2.dropna(axis=1, how='all').to_csv(fr'{path_raw}\MRH2.csv')

if __name__ == '__main__':
    # seperate()
    MRH_questionnaire_1()
