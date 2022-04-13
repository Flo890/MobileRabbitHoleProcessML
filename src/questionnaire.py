import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    p = df_MRH1.SD01.value_counts().sort_index(axis=0).plot(kind = 'bar')

    plt.show()



def seperate():
    df_qu = pd.read_csv(path_questionnaire,  sep=',')
    df_MRH1 = df_qu[df_qu['QUESTNNR'] == 'MRH1']
    df_MRH2 = df_qu[df_qu['QUESTNNR'] == 'MRH2']

    df_MRH1.dropna(axis=1, how='all').to_csv(fr'{path_raw}\MRH1.csv')
    df_MRH2.dropna(axis=1, how='all').to_csv(fr'{path_raw}\MRH2.csv')

if __name__ == '__main__':
    # seperate()
    MRH_questionnaire_1()
