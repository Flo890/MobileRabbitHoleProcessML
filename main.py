# This is a sample Python script.
import datetime
import json
import os
import re

import pandas as pd


def import_json():
    dfs = []  # an empty list to store the data frames

    # path_to_json = 'M:/+Dokumente/+Studium/11. Semester/Masterarbeit/RabbitHoleTrackerData/toUse/*'
    data_path = r'/toUse/data.json'
    data_pathtest = r'.\cleanFiles\xxxxmm\xxxxmm_logs_2022_03_26_11_41_45.json'
    data_path = r'M:\+Dokumente\PycharmProjects\RabbitHolePreprocess\cleanFiles\xxxxmm\xxxxmm_logs_2022_03_26_11_41_45.json'

    with open(data_path, encoding='utf-8') as data:
        d = json.load(data)

    # us = pd.json_normalize(jd["users"])
    # print(us.head())
    # data = jd['hits']['hits']
    #  dict_flattened = (flatten(record, '.') for record in jd)
    # df = pd.DataFrame(dict_flattened)
    # print(df)

    logs = pd.read_json(data_path, orient="index")
    print(logs)

    # fill Nan
    # logs['metaData'].fillna('{}', inplace=True)
    logs_meta = pd.json_normalize(logs['metaData']) # record path??
    print(logs_meta)
    logs.drop(columns=['metaData'])

    logs_final = logs.join(logs_meta)
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
    data_path = r'/toUse/data.json'

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
