"""
Created on Wed Sep 22 14:44:25 2021

@author: annachiararossi
@author: alessandrocogliati
"""

import pandas as pd

filename = 'data.csv'
df = pd.read_csv(filename, sep=';')  # gives a dataframe
# raw_data = df.values            # convert to array


### first remove the most extreme outliers

df = df[df['time'] < 3.0]
df = df.reset_index(drop=True)

### standardize for each person
from scipy import stats

for elem_id in df['id'].unique():
    df.loc[df['id'] == elem_id, 'time'] = stats.zscore(df.loc[df['id'] == elem_id, 'time'], ddof=1)

### convert to a continuous index

from sklearn.preprocessing import MinMaxScaler

## put Non-smiling in the range [-1,0), and the most negative value is the fastest

mask_not_smiling = df.label == 'Not smiling'
df.loc[mask_not_smiling, 'time'] = MinMaxScaler(feature_range=(-1, 0)).fit_transform(df[mask_not_smiling]['time'].values.reshape(-1, 1))

## put Smiling times in the range (0,1], switching the values to get high values for low reaction times
mask_smiling = df.label == 'Smiling'
df.loc[mask_smiling, 'time'] = - df.loc[mask_smiling, 'time']
df.loc[mask_smiling, 'time'] = MinMaxScaler(feature_range=(0, 1)).fit_transform(df[mask_smiling]['time'].values.reshape(-1, 1))

# ### THRESHOLD, NOT DOING IT
# m_s = (df.loc[mask_smiling, 'time']).mean()
#
# # mean for both labels
# m_ns = (df.loc[mask_not_smiling, 'time']).mean()
#
# # set to -1 the ones below the mean
#
# df_not_smiling = df[mask_not_smiling]['time']
# mask_inside_not_smiling = df_not_smiling < m_ns
#
# import numpy as np
# df_not_smiling = np.where(mask_inside_not_smiling, -1, df_not_smiling.values)
# df_not_smiling = MinMaxScaler(feature_range=(-1, 0)).fit_transform(df_not_smiling.reshape(-1, 1))
# df.loc[mask_not_smiling, 'time'] = df_not_smiling
#
# # set to 1 the ones above the mean
# if df[mask, 'time'] > m_s:
#     df.loc[mask, 'time'] = 1

## now all the values are rescaled in the correct ranges


# readjust the dataset to get for each image the 4 indeces (one for each test person)

# since we remove some outliers (which were not outliers for all test people)
# we are going to get that some images have less indeces
# so we should be careful and understand what we should do about that

# the smiling not-smiling is removing the id of the image --> not good!!
