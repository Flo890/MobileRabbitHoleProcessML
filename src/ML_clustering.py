import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from numpy import where, unique
from sklearn.cluster import KMeans
import ML_helpers
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.rcParams['figure.dpi'] = 200

dataframe_dir_ml = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\ML'

def dcbscan(x):
    x = np.array(x)
    model = DBSCAN(eps=0.30, min_samples=9)
    # fit model and predict clusters
    yhat = model.fit_predict(x)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(x[row_ix, 0], x[row_ix, 1])
    # show the plot
    plt.show()

def k_means(x):
    print("----k-means----")
    x = np.array(x, dtype=object)
    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit(x) # kmeans.fit_predict(x)
    print(kmeans.labels_[:10])
    print(type(x))
    plt.scatter(x[:, 0], x[:, 1])
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.show()



def k_means_determine_cluster(x):
    print("-------k-means determine clusters--------")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    # Optimum at k = 4

def prepare_clustering(df):
    return df[['f_session_length', 'f_esm_more_than_intention_Yes']].fillna(0)

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    # check_labels()

    df_sessions = pd.read_pickle(fr'{dataframe_dir_ml}\user-sessions_features_all.pickle')

    x_prep = prepare_clustering(df_sessions)
    # k_means(x_prep)
    dcbscan(x_prep)
