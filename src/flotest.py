import pandas as pd
import featureExtraction

if __name__ == '__main__':

    ## extracts features for one session
    df_logs = pd.read_pickle(        "D:\\usersorted_logs_preprocessed\\AN09BI.pickle")#AN23GE.pickle")
    df_all_sessions = pd.read_pickle("C:\\projects\\rabbithole\\RabbitHoleProcess\\data\\dataframes\\sessions_raw_extracted\\AN09BI-sessions.pickle")

    df_session_features = featureExtraction.get_features_for_session(df_logs=df_logs,
                                                                     df_sessions=df_all_sessions)
