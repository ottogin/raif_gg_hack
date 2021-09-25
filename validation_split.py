import pandas as pd

train = pd.read_csv('data/train.csv')
train['parsed_date'] = pd.to_datetime(train.date)
train = train.sort_values(by='parsed_date')
n = len(train)
val_size = int(round(n * 0.1))
train_set, val_set = train.iloc[:n - val_size], train.iloc[n - val_size:]

train_set.drop('parsed_date', axis=1).to_csv('data/train_trunc.csv', index=False)
val_set.drop('parsed_date', axis=1).to_csv('data/validation.csv', index=False)
