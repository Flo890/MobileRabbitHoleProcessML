import datetime
import json
import pathlib
import re
import gzip
import pandas as pd
import pickle


class CleanJson:
    def __init__(self):
        self.message = 'cleanData'

    def extractMetaData(self, logs):
        """
        extract the metaData from json child logs
        :param logs: the dataframe with the colum of metaData to extract
        """
        print("Extracting MetaData")
        # fill Nan
        # logs['metaData'].fillna('{}', inplace=True)

        # Normalize the metaData
        logs_meta = pd.json_normalize(logs['metaData'])  # record path??
        logstest = pd.DataFrame.from_dict(logs['metaData'], orient="index")
        # Merge metaData and logs
        logs_final = pd.concat([logs.reset_index(drop=True), logs_meta.reset_index(drop=True)], axis=1)
        return logs_final

    def extract_logs(self, directory, end_directory, save_type, is_gzip):
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

        logs_dic = {}
        for data_path in pathlist:
            # data_path = one day of logs

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
                    stud_id = self.getStudyID(json_data['users'][key]['account_email'])
                    temp_user = json_data['users'][key]['logs']
                    print(f"get logs from {stud_id}")

                    # create a dataframe form the json logs
                    df = pd.DataFrame.from_dict(temp_user, orient="index")

                    if stud_id not in logs_dic:
                        logs_dic[stud_id] = []

                    # save the logs of current file to user dic
                    logs_dic[stud_id].append(df)

                except(Exception,):
                    print(f"error extracting logs for {key}")
                    continue

        temp_all_Logs = []
        print("_________________________")
        for key in logs_dic.keys():
            try:
                df_concat = pd.concat(logs_dic[key], ignore_index=False)
                # add a colum with the studyId to each log
                df_concat["studyID"] = key
                logs_dic[key] = df_concat

                if save_type == 1:
                    temp_all_Logs.append(df_concat)
                elif save_type == 3:
                    # save all logs from one user to file
                    output_users_name_dic = fr"{end_directory}\{key}_logs.pickle"
                    with open(output_users_name_dic, 'wb') as f:
                        # Pickle the 'data' dictionary using the highest protocol available.
                        pickle.dump(logs_dic[key], f, pickle.HIGHEST_PROTOCOL)

                print(f"saved logs for {key}")

            except(Exception,):
                print(f"error concat logs for {key}")
                continue

        if save_type == 1:
            # save all logs in one giant dataframe
            all_logs = pd.concat(temp_all_Logs, ignore_index=False)
            output_users_name_all = fr"{end_directory}\+_logs_all.pickle"
            with open(output_users_name_all, 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(all_logs, f, pickle.HIGHEST_PROTOCOL)

        elif save_type == 2:
            # save all logs sorted by user
            output_users_name_dic = fr"{end_directory}\+_logs_dic.pickle"
            with open(output_users_name_dic, 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(logs_dic, f, pickle.HIGHEST_PROTOCOL)

        print("finished extrating.")

    def extract_logs_different(self, directory, end_directory):
        """
        Extracts for all json files in a directory the logs for each user and stores them in seperate files.
        :param directory: the directory where all json files to extract are stored
        :param end_directory: the directory where the extracted logs will be stored
        """

        #other idea: delete everything exept for logs? and then read json?
        # use dask


        print("Extracting Logs")
        # for raw json
        # pathlist = pathlib.Path(directory).glob('**/*.json')
        # for gzip
        pathlist = pathlib.Path(directory).glob('**/*.gz')

        logs_dic = {}
        for data_path in pathlist:
            # data_path = one day of logs

            path_in_str = str(data_path)
            print(f"extract: {path_in_str}")

            data = pd.read_json(path_in_str, orient="index")
            print(data.head(10))
            print("-------------------------")
            # read howle json iterate throgh?

            # dic = pd.DataFrame()
            # for chunk in pd.read_json(path_in_str, orient="index", lines = True, chunksize=10000):
            #    dic = dic.append(chunk)
            # df = pd.concat(chunks)
            # print(dic.head())

            logs_dic = {}
            user_dic = {}
            for key in data:
                # try:
                user_dataframe = data[key].apply(pd.Series)
                print("_______________________")
                print(user_dataframe.head())
                print(user_dataframe.columns.values)
                print("....._")
                stud_id = self.getStudyID(user_dataframe['account_email'])

                logs = pd.json_normalize(user_dataframe["logs"], max_level=0)  # user_dataframe["logs"].apply(pd.Series)
                logs = user_dataframe["logs"].apply(pd.Series)
                print(logs.head())
                print(logs.columns.values)
                print("_______________________")
                # if stud_id not in user_dic:
                #    user_dic[stud_id] = []
                #    logs_dic[stud_id] = []

                # TODO make sure to not add email etc suplicate? save intentions extrac?
                # logs_dic[stud_id].append(df)

            # except:
            #     print(f"error extracting logs for {key}")
            #     continue

        print("finished extrating.")

    def extractUsers(self, file_path):
        print("Extracting User")

        # with open(file_path, encoding='utf-8') as data:
        #     json_data = json.load(data)

        with gzip.open(file_path, 'r') as data:
            json_data = json.loads(data.read().decode('utf-8'))

        users = json_data['users']

        # save logs in dic and delete from cleaned file.
        for key in users:
            try:
                temp_user = json_data['users'][key]
                del temp_user['logs']

            except:
                continue

        # save the users
        output_users_name = "M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\cleanedUsers\cleanedUsers.json"
        with open(output_users_name, "w+") as file:
            json.dump(json_data, file, sort_keys=True)

    def getStudyID(self, studyID_email):
        study_id = re.sub('@email\.com$', '', str(studyID_email))
        return study_id


if __name__ == '__main__':
    raw_data_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\live'
    raw_data_user = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\live\2022-03-25T00 44 32Z_1Uhr46absentmindedtrack-default-rtdb_data.json.gz'
    logs_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\logFiles'
    user_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\cleanedUsers'
    dataframe_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes'

    # extract all logs for each user
    cleanJson = CleanJson()
    # cleanJson.extractUsers(raw_data_user)
    # cleanJson.extract_logs(directory=raw_data_dir, end_directory=logs_dir,  save_as_one=False)
    cleanJson.extract_logs_different(directory=raw_data_dir, end_directory=dataframe_dir, )
