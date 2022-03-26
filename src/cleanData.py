# This is a sample Python script.

# This is a sample Python script.
import datetime
import json
import os
import re

import pandas as pd


def import_json():
    dfs = []  # an empty list to store the data frames

    # path_to_json = 'M:/+Dokumente/+Studium/11. Semester/Masterarbeit/RabbitHoleTrackerData/toUse/*'
    data_path = r'M:\+Dokumente\PycharmProjects\toUse\data.json'
    data_pathtest = r'M:\+Dokumente\PycharmProjects\cleanFiles\xxxxmm\xxxxmm_logs_2022_03_25_13_41_18.json'

    with open(data_pathtest, encoding='utf-8') as data:
        d = json.load(data)

    # us = pd.json_normalize(jd["users"])
    # print(us.head())
    # data = jd['hits']['hits']
    #  dict_flattened = (flatten(record, '.') for record in jd)
    # df = pd.DataFrame(dict_flattened)
    # print(df)

    logs = pd.read_json(data_pathtest, orient="index")
    print(logs)

    # fill Nan
    # logs['metaData'].fillna('{}', inplace=True)
    logs_meta = pd.json_normalize(logs['metaData'])
    logs.drop(columns=['metaData'])

    logs_final = logs.join(logs_meta)
    logs.drop(columns=['metaData'])
    print(logs_final.columns.tolist())
    print(logs_final)

    # Convert strings to JSON objects
    # df['readings'] = df['readings'].map(json.loads)

    # Can't use nested lists of JSON objects in pd.json_normalize
    # df = logs.explode(column='metaData').reset_index(drop=True)

    # logs.drop(columns=['metaData'])
    # logs = pd.concat([logs[['eventName']], pd.json_normalize(logs['metaData'])])
    # logs.drop(columns=['dataKey'])
    # print(logs)
    # print(logs.columns.tolist())

    # d = logs.to_dict(orient='records')
    # df = pd.json_normalize(d, record_path=['line_items'], meta=['order_id', 'email'])

    # df1 = (pd.concat({i: pd.json_normalize(x) for i, x in logs.pop('metaData').items()})
    #        .reset_index(level=1, drop=True)
    #        .join(logs)
    #        .reset_index(drop=True))


    # Target_df = pd.concat([json_normalize(source_df['COLUMN'][key], 'volumes', ['name','id','state','nodes'], record_prefix='volume_') for key in source_df.index]).reset_index(drop=True)

    #  df_final = pd.json_normalize(json.loads(logs['metaData']))
    # print(metan)
    # print(df_final)


#  print(logs)

#  print(logs.columns.tolist())
# except:
# print("error")


# d["data"] = [d["data"][k] for k in d["data"].keys()]
# pd.json_normalize(pd.json_normalize(d).explode("data").to_dict(orient="records"))

# users = jd['users']
# for key in users:
# us = pd.json_normalize(jd[key])
# print(us.head())


def cleanFiles():
    data_path = r'M:\+Dokumente\PycharmProjects\toUse\data.json'

    with open(data_path, encoding='utf-8') as data:
        jsonData = json.load(data)

    users = jsonData['users']
    user_list_json = []
    loglist = {}
    currentDate = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    print(currentDate)

    # save logs in dic and delete from cleaned file.
    for key in users:
        try:
            temp_user = jsonData['users'][key]
            stud_id = getStudyID(temp_user['account_email'])
            loglist[stud_id] = temp_user['logs']

            filename = f".\cleanFiles\{stud_id}"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(f'{filename}\{stud_id}_logs_{currentDate}.json', "w+") as file:
                json.dump(temp_user['logs'], file, sort_keys=True, indent=4)

            del temp_user['logs']

        except:
            continue

    # # save the logs
    # output_logs_name = ".\cleanFiles\cleanedLogs.json"
    # with open(output_logs_name, "w+") as file:
    #     json.dump(loglist, file, sort_keys=True, indent=4)

    # save the users
    output_users_name = ".\cleanFiles\cleanedUsers.json"
    with open(output_users_name, "w+") as file:
        json.dump(jsonData, file, sort_keys=True, indent=4)


def getStudyID(studyID_email):
    study_id = re.sub('@email\.com$', '', studyID_email)
    print(study_id)
    return study_id


# cleanFiles()
import_json()
# importFiles()

# def test():
# extract columes
# new = old[['A', 'C', 'D']].copy()

# nested json object:
#  FIELDS = ["key", "fields.summary", "fields.issuetype.name", "fields.status.name", "fields.status.statusCategory.name"]
# df = pd.json_normalize(results["issues"])
# df[FIELDS]
# Only recurse down to the second level
# pd.json_normalize(results, record_path="issues", max_level =


# _____________________________________________________________________________________________________________________________
# if __name__ == '__main__':
#     import_json()

def import_json():
    dfs = []  # an empty list to store the data frames
    temp = pd.DataFrame()

    # path_to_json = 'M:/+Dokumente/+Studium/11. Semester/Masterarbeit/RabbitHoleTrackerData/toUse/*'
    dataPath = r'M:\+Dokumente\PycharmProjects\toUse\data.json'

    with open(dataPath, encoding='utf-8') as data:
        jsonData = json.load(data)

    # pd.read_json (r'M:\+Dokumente\PycharmProjects\data.json')

    # json_pattern = os.path.join(path_to_json, '*.json')
    # file_list = glob.glob(json_pattern)
    data = pd.read_json(path_to_file)
    # read compressed
    # df_gzip = pd.read_json('sample_file.gz', compression='infer')

    # data_n = pd.json_normalize(path_to_file)
    print(data.head())
    # print(data_n.head())

    # for file in file_list:
    #     data = pd.read_json(file, lines=True)  # read data frame from json file
    #     dfs.append(data)  # append the data frame to the list
    # temp = pd.concat(dfs, ignore_index=True)  # concatenate all the data frames in the list.
    # dfs[0].head()


# def importFiles1():
#     dataPath = r'M:\+Dokumente\PycharmProjects\toUse\data.json'
#     with open(dataPath, encoding='utf-8') as data:
#         jsonData = json.load(data)
#
#     data = pd.read_json(r'M:\+Dokumente\PycharmProjects\toUse\data.json')
#
#     users = jsonData['users']
#     user_list_json = []
#     user_list_df = []
#
#     for key in users:
#         try:
#             temp_user = jsonData['users'][key]
#             user_list_json.append(temp_user)
#         except:
#             continue
#
#     # for user in user_list_json:
#     #     data = pd.read_json(user)
#     #     user_list_df.append(data)
#

def cleanFiles():
    dataPath = r'M:\+Dokumente\PycharmProjects\toUse\data.json'

    with open(dataPath, encoding='utf-8') as data:
        jsonData = json.load(data)

    users = jsonData['users']
    user_list_json = []
    loglist = {}
    currentDate = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    print(currentDate)

    # save logs in dic and delete from cleaned file.
    for key in users:
        try:
            temp_user = jsonData['users'][key]
            stud_id = getStudyID(temp_user['account_email'])
            loglist[stud_id] = temp_user['logs']

            filename = f".\cleanFiles\{stud_id}"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(f'{filename}\{stud_id}_logs_{currentDate}.json', "w+") as file:
                json.dump(temp_user['logs'], file, sort_keys=True, indent=4)

            del temp_user['logs']

        except:
            continue

    # # save the logs
    # output_logs_name = ".\cleanFiles\cleanedLogs.json"
    # with open(output_logs_name, "w+") as file:
    #     json.dump(loglist, file, sort_keys=True, indent=4)

    # save the users
    output_users_name = ".\cleanFiles\cleanedUsers.json"
    with open(output_users_name, "w+") as file:
        json.dump(jsonData, file, sort_keys=True, indent=4)


def getStudyID(studyID_email):
    study_id = re.sub('@email\.com$', '', studyID_email)
    print(study_id)
    return study_id


cleanFiles()
# importFiles()

# def test():
# extract columes
# new = old[['A', 'C', 'D']].copy()

# nested json object:
#  FIELDS = ["key", "fields.summary", "fields.issuetype.name", "fields.status.name", "fields.status.statusCategory.name"]
# df = pd.json_normalize(results["issues"])
# df[FIELDS]
# Only recurse down to the second level
# pd.json_normalize(results, record_path="issues", max_level =
