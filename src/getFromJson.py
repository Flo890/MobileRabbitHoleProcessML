import json
import pathlib
import re
import gzip
import pandas as pd
import pickle


def extractMetaData(df_logs):
    """
    extract the metaData from json child logs
    :param logs: the dataframe with the colum of metaData to extract
    """
    print("_Extracting MetaData_")

    if 'metaData' in df_logs.columns:
        # df_logs['metaData'].fillna('{}', inplace=True)
        # Normalize the metaData
        logs_meta = pd.json_normalize(df_logs['metaData'])  # record path??
        # Merge metaData and logs
        df_logs.drop(columns=['metaData'], inplace=True)
        logs_final = pd.concat([df_logs.reset_index(drop=False), logs_meta.reset_index(drop=True)], axis=1)
        return logs_final
        # logs_final.to_csv(fr'{dataframe_dir}\test2.csv')
    else:
        return df_logs


def extract_logs(directory, end_directory, is_gzip):
    """
    Extracts for all json files in a directory the logs for each user and stores them in seperate files.
    :param directory: the directory where all json files to extract are stored
    :param end_directory: the directory where the extracted logs will be stored
    :param save_type: specify if the logs should be stored:
        1 for all in one df and file
        2 for all in one dic file, user-sorted
        3 for each user get extra file
    :param is_gzip: specify if the files to read are gzip (true) or json(false)
    """
    print("Extracting Logs")

    if is_gzip:
        # for gzip
        pathlist = pathlib.Path(directory).glob('**/*.gz')
    else:
        # for raw json
        pathlist = pathlib.Path(directory).glob('**/*.json')

    # logs_dic = {}
    for data_path in pathlist:
        # data_path = one day of logs
        logs_dic = {}

        path_in_str = str(data_path)
        print(f"extract: {path_in_str}")

        if is_gzip:
            # for gzip
            with gzip.open(path_in_str, 'r') as data:
                json_data = json.loads(data.read().decode('utf-8'))
        else:
            # for raw json
            with open(path_in_str, encoding='utf-8') as data:
                json_data = json.load(data)

        users = json_data['users']
        #  currentDate = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        for key in users:
            try:
                # Find the studyId
                stud_id = getStudyID(json_data['users'][key]['account_email'])
                print(f"get logs from {stud_id}")

                # if stud_id not in logs_dic:
                #     logs_dic[stud_id] =

                # save the logs of current file to user dic
                logs_dic[stud_id] = pd.DataFrame.from_dict(json_data['users'][key]['logs'], orient="index")

            except(Exception,):
                print(f"error extracting logs for {key}")
                continue

        output_users_name_dic = fr"{end_directory}\logs_dic_{data_path.stem}.pickle"
        with open(output_users_name_dic, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(logs_dic, f, pickle.HIGHEST_PROTOCOL)

    print("finished extrating.")


def merge_two_dicts(x, y):
    z = x.copy()  # start with keys and values of x
    z.update(y)  # modifies z with keys and values of y
    return z


def concat_logs_one_user(logs_dic_directory, end_directory, studyID):
    print("_____ concatUser _____")
    pathlist = pathlib.Path(logs_dic_directory).glob('**/*.pickle')

    # logs_dic = pd.DataFrame()
    logs_dic_0 = {}
    for id in studyID:
        logs_dic_0[id] = []

    for data_path in pathlist:
        path_in_str = str(data_path)
        print(f"read dic: {path_in_str}")
        with open(path_in_str, 'rb') as file:
            read_logs_dic = pickle.load(file)

        for id in studyID:
            index = studyID.index(id)
            if studyID[index] in read_logs_dic:
                print(studyID[index], 'append')
                # logs_dic = pd.concat([logs_dic, read_logs_dic[studyID]], ignore_index=False)
                logs_dic_0[id].append(read_logs_dic[studyID[index]])
            else:
                print(studyID[index], 'not append')

    try:
        for id in studyID:
            index = studyID.index(id)
            if logs_dic_0[id]:
                print('save', studyID[index])
                df = pd.concat(logs_dic_0[id])
                df["studyID"] = studyID[index]
                output_users_name_dic = fr"{end_directory}\{studyID[index]}.pickle"
                with open(output_users_name_dic, 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)
    except:
        print('no concat')

    print(f"saved logs for {studyID}")




def concat_user_logs(logs_dic_directory, end_directory):
    print("_____ concatUser _____")
    pathlist = pathlib.Path(logs_dic_directory).glob('**/*.pickle')

    logs_dic = {}
    for data_path in pathlist:
        path_in_str = str(data_path)
        print(f"read dic: {path_in_str}")
        with open(path_in_str, 'rb') as file:
            read_logs_dic = pickle.load(file)

        print(read_logs_dic.keys())

        # logs_dic = merge_two_dicts(logs_dic, read_logs_dic)
        for key in read_logs_dic.keys():
            if key not in logs_dic:
                logs_dic[key] = read_logs_dic[key]
            else:
                print('read ', type(read_logs_dic[key]))
                logs_dic[key] = pd.concat([logs_dic[key], read_logs_dic[key]], ignore_index=False)
                print(type(logs_dic[key]))
                print(key, len(logs_dic[key]))
            # df_concat = pd.concat(logs_dic[key], ignore_index=False)
            # logs_dic[key] =

    print('concat')
    for key in logs_dic.keys():
        #  try:
        # print(type(logs_dic[key]))
        # print(logs_dic[key])
        # df_concat = pd.concat(logs_dic[key], ignore_index=False)
        # add a colum with the studyId to each log
        logs_dic[key]["studyID"] = key

        # if save_type == 1:
        #     temp_all_Logs.append(df_concat)
        # elif save_type == 3:
        # save all logs from one user to file
        output_users_name_dic = fr"{end_directory}\{key}.pickle"
        with open(output_users_name_dic, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(logs_dic[key], f, pickle.HIGHEST_PROTOCOL)

        print(f"saved logs for {key}")

    # except(Exception,):
    #    print(f"error concat logs for {key}")
    #    continue

    # if save_type == 1:
    #     # save all logs in one giant dataframe
    #     all_logs = pd.concat(temp_all_Logs, ignore_index=False)
    #     output_users_name_all = fr"{end_directory}\+_logs_all.pickle"
    #     with open(output_users_name_all, 'wb') as f:
    #         # Pickle the 'data' dictionary using the highest protocol available.
    #         pickle.dump(all_logs, f, pickle.HIGHEST_PROTOCOL)
    #
    # elif save_type == 2:
    #     # save all logs sorted by user
    #     output_users_name_dic = fr"{end_directory}\+_logs_dic.pickle"
    #     with open(output_users_name_dic, 'wb') as f:
    #         # Pickle the 'data' dictionary using the highest protocol available.
    #         pickle.dump(logs_dic, f, pickle.HIGHEST_PROTOCOL)


def extractUsers(file_path, end_directory, is_gzip):
    """
    Extract all users without the logs
    :param file_path:
    :param end_directory:
    :param is_gzip:
    :return:
    """
    print("Extracting User")

    if is_gzip:
        with gzip.open(file_path, 'r') as data:
            json_data = json.loads(data.read().decode('utf-8'))
    else:
        with open(file_path, encoding='utf-8') as data:
            json_data = json.load(data)

    users = json_data['users']

    # save logs in dic and delete from cleaned file.
    for key in users:
        try:
            temp_user = json_data['users'][key]
            del temp_user['logs']

        except:
            continue

    # save the users as dataframe
    df_users = pd.DataFrame.from_dict(json_data)
    df_users_n = pd.json_normalize(df_users['users'], max_level=0)
    df_users.drop(columns=['users'], inplace=True)
    users_final = pd.concat([df_users.reset_index(drop=True), df_users_n.reset_index(drop=True)], axis=1)

    users_final.to_csv(fr'{end_directory}\cleand_users.csv')

    output_users_name = fr"{end_directory}\cleaned_users.pickle"
    with open(output_users_name, "wb") as f:
        pickle.dump(users_final, f, pickle.HIGHEST_PROTOCOL)

    return users_final


def getStudyID(studyID_email):
    study_id = re.sub('@email\.com$', '', str(studyID_email))
    return study_id


def getStudIDlist(path_cleaned_users):
    cleand_users = pd.read_pickle(path_cleaned_users)
    return cleand_users['account_email'].apply(getStudyID).tolist()


if __name__ == '__main__':
    raw_data_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\live'
    #     raw_data_user = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\live\2022-03-25T00 44 32Z_1Uhr46absentmindedtrack-default-rtdb_data.json.gz'
    logs_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\logFiles'
    #     user_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\cleanedUsers'
    #     dataframe_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes'
    clean_users_path = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes\users\cleaned_users.pickle'
    r = getStudIDlist(clean_users_path)
    print(r)
#
#     # extractUsers(raw_data_user)
    extract_logs(directory=raw_data_dir, end_directory=logs_dir, save_type=3)
