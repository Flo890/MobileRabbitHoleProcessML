# Get dataframe with all Screen events (also shutdown?)

# get all timestamps 1: User ON_USERPRESENT  (SCREEN_ON_UNLOCKED?)

# While all ts1 are not nan

# find ts2: OFF_LOCKED, OFF_UNLOCKED, everything else that is not ON_USerpresent? und timestamp davon is later than tsi
# if tss is nan, take max timestamp of that data

# calculate length of session

# assign sessionID: count and timestamp?

#get all data where timestamp is smaller thatn ts1 but larger than ts 1
# assign sessionID to all these rows with sessionid

# add a new row to sessiondf with sessionid, sessionlngth, first and last timestamp

# increment count

# get the new ts1 where timestamp is larger than ts2




# OR:
# Group with session id, loop over groups and do steps above (with exption that USER_present is dumbly in session previously
