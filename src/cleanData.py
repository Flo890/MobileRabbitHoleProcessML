import datetime
import json
import pathlib
import re

import pandas as pd


class CleanData:
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
        pathlist = pathlib.Path(directory).glob('**/*.json')
        for data_path in pathlist:
            # because path is object not string
            path_in_str = str(data_path)
            print(f"extract: {path_in_str}")
            with open(path_in_str, encoding='utf-8') as data:
                json_data = json.load(data)

            users = json_data['users']
            # loglist = {} # needed when puting all logs in one file
            currentDate = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

            # save logs in dic and delete from cleaned file.
            for key in users:
                try:
                    temp_user = json_data['users'][key]
                    stud_id = self.getStudyID(temp_user['account_email'])
                    print(stud_id)
                    logs = temp_user['logs']

                    dirname = f"{end_directory}\{stud_id}"
                    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
                    # filename = f'{dirname}\{stud_id}_logs_{currentDate}.json'
                    filename = f'{dirname}\{stud_id}_logs_{data_path.stem}_{currentDate}.json'
                    with open(filename, "w+") as file:
                        json.dump(json_data['users'][key], file, sort_keys=True)

                    del temp_user['logs']

                except:
                    print(f"error extracting logs for {key}")
                    continue
        print("finished extrating.")

    def extractUsers(self, file_path):
        print("Extracting User")
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

        # save the users
        output_users_name = ".\cleanFiles\cleanedUsers.json"
        with open(output_users_name, "w+") as file:
            json.dump(json_data, file, sort_keys=True)

    def getStudyID(self, studyID_email):
        study_id = re.sub('@email\.com$', '', studyID_email)
        return study_id
