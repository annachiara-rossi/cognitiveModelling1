"""
Created on Wed Sep 22 14:44:25 2021

@author: annachiararossi
"""

import pandas as pd

filename = '/Users/annachiararossi/Documents/Cognitive Modeling/report1/data.csv'
df = pd.read_csv(filename, sep=';')  # gives a dataframe
# raw_data = df.values            # convert to array


### first remove the most extreme outliers

df = df[df['time'] < 3.0]
df = df.reset_index(drop=True)

### standardize for each person

from scipy import stats

for id in df['id'].unique():
    df.loc[df['id'] == id, 'time'] = stats.zscore(df.loc[df['id'] == id, 'time'], ddof=1)

### convert to a continuous index

from sklearn.preprocessing import MinMaxScaler

## put Non-smiling in the range [-1,0), and the most negative value is the fastest

mask = df.label == 'Not smiling'
df.loc[mask, 'time'] = MinMaxScaler(feature_range=(-1, 0)).fit_transform(df[mask]['time'].values.reshape(-1, 1))
# mean
m_ns = (df.loc[mask, 'time']).mean()

# set to -1 the ones below the mean
df.loc[df[mask, 'time'] < m_ns] = -1
df.loc[mask, df['time'] < m_ns] = -1

df[(df.label == 'Not Smiling') & (df.time < m_ns)]

## put Smiling times in the range (0,1], switching the values to get high values for low reaction times
mask = df.label == 'Smiling'
df.loc[mask, 'time'] = - df.loc[mask, 'time']
df.loc[mask, 'time'] = MinMaxScaler(feature_range=(0, 1)).fit_transform(df[mask]['time'].values.reshape(-1, 1))
m_s = (df.loc[mask, 'time']).mean()

# set to 1 the ones above the mean
if df[mask, 'time'] > m_s:
    df.loc[mask, 'time'] = 1

## now all the values are rescaled in the correct ranges


# readjust the dataset to get for each image the 4 indeces (one for each test person)

# since we remove some outliers (which were not outliers for all test people)
# we are going to get that some images have less indeces
# so we should be careful and understand what we should do about that

# the smiling not-smiling is removing the id of the image --> not good!!
