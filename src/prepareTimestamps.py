# Do this all for one user at a time?

# 1. Add timezoneOffset to timestamps
# 2. convert milliseconds to date format
# 3. order rows to logging timestamp
# 4. add further interesting variables etc weekday

def process_timestamps(df_logs):
    # 1. Add timezoneOffset to timestamps
    df_logs["correct_timestamp"] = df_logs["timestamp"]
    logs['timestamp'] = pd.to_datetime(logs['timestamp'], yearfirst=True, unit='ms')
