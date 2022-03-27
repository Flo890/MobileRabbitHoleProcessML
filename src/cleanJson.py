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
        # Merge metaData and logs
        logs_final = pd.concat([logs.reset_index(drop=True), logs_meta.reset_index(drop=True)], axis=1)
        return logs_final

    def extractLogs(self, directory, end_directory):
        """
        Extracts for all json files in a directory the logs for each user and stores them in seperate files.
        :param directory: the directory where all json files to extract are stored
        :param end_directory: the directory where the extracted logs will be stored
        """
        print("Extracting Logs")
        # for raw json
        pathlist = pathlib.Path(directory).glob('**/*.json')
        # for gzip
        # pathlist = pathlib.Path(directory).glob('**/*.gz')

        logs_dic = {}
        for data_path in pathlist:
            # data_path = one day of logs

            path_in_str = str(data_path)
            print(f"extract: {path_in_str}")

            # for raw json
            with open(path_in_str, encoding='utf-8') as data:
                json_data = json.load(data)
            # for gzip
            # with gzip.open(path_in_str, 'r') as data:
            #     json_data = json.loads(data.read().decode('utf-8'))

            users = json_data['users']
            # loglist = {} # needed when puting all logs in one file
            currentDate = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

            for key in users:
                try:
                    stud_id = self.getStudyID(json_data['users'][key]['account_email'])
                    temp_user = json_data['users'][key]['logs']
                    print(stud_id)

                    # dirname = f"{end_directory}\{stud_id}"
                    # pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
                    # filename = f'{dirname}\{stud_id}_logs_{currentDate}.json'
                    # filename = f'{dirname}\{stud_id}_logs_{data_path.stem}_{currentDate}.json'
                    # with open(filename, "w+") as file:
                    #     json.dump(temp_user['logs'], file, sort_keys=True)
                    df = pd.DataFrame.from_dict(temp_user, orient="index")

                    if stud_id not in logs_dic:
                        print(stud_id + "not in dic")
                        logs_dic[stud_id] = []
                    print(logs_dic[stud_id])
                    logs_dic[stud_id] = logs_dic[stud_id].append(df)

                except:
                    print(f"error extracting logs for {key}")
                    continue


        for key in logs_dic.keys():
            print(logs_dic[key])
            # logs_dic[key] = pd.concat(logs_dic[key], ignore_index=False)

        output_users_name = "M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\logFiles\logs.pickle"
        with open(output_users_name, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(logs_dic, f, pickle.HIGHEST_PROTOCOL)
        print(f"saved:{len(logs_dic)}")
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
        study_id = re.sub('@email\.com$', '', studyID_email)
        return study_id


if __name__ == '__main__':
    raw_data_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\test'
    raw_data_user = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\rawData\live\2022-03-25T00 44 32Z_1Uhr46absentmindedtrack-default-rtdb_data.json.gz'
    logs_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\logFiles'
    user_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\cleanedUsers'
    dataframe_dir = r'M:\+Dokumente\PycharmProjects\RabbitHoleProcess\data\dataframes'

    # extract all logs for each user
    cleanJson = CleanJson()
    # cleanJson.extractUsers(raw_data_user)
    cleanJson.extractLogs(directory=raw_data_dir, end_directory=logs_dir)
