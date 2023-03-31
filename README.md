# MobileRabbitHoleProcessML

## Preprocessing & Feature Generation

`featureGenerationGroup.py`

In its main method one can try the feature extraction for a single user
- takes Miriam's non-grouped sessions (RabbitHoleProcess\\data\\dataframes\\sessions_with_features\\*.pickle) as basis
- sessions are clustered into session-groups (methods imported from grouped_sessions.py)

This file implements the methods that actually create features, however its main method is just for test and dev purposes

(old: featureGeneration.py - the stuff Miriam did)

`preprocessingGroup.py`

Actually performs the preprocessing (if desired) for all users.

TODO input output?

`ML_classification.py`

TODO

## Data

(with exemplary user AN23GE)

### Miriam's features of non-grouped sessions
`RabbitHoleProcess\\data\\dataframes\\sessions_with_features\\AN23GE.pickle`
One line per unlock/lock session, one column per generated feature. Includes columns for the ESM responses

### All Raw Sensing Data
`C:\\Users\\florianb\\Downloads\\AN23GE.pickle` <- one example
`D:\usersorted_logs_preprocessed\\AN23GE.pickle` <- all
One line per sensing event log (further data on Flo's external HDD due to lack of storage)


## MobileHCI'23 Workflow:

- run preprocessing.py (uncomment the stuff you want in the bottom of the file)
- run gridsearch-sessions.py

The DescriptiveStatsNew.py file might be used for some descriptive analyses