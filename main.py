"""
Created on Wed Sep 22 14:44:25 2021

@author: annachiararossi
@author: alessandrocogliati
"""

import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import os
import matplotlib.image as mpimg

filename = 'data.csv'
df = pd.read_csv(filename, sep=';')  # gives a dataframe
# raw_data = df.values            # convert to array

# create a set contianing the number of the image which is an outlier for at least one test person
outlier_mask = df['time'] >= 3.0

sum = 0
out_list = []
for i in range(len(outlier_mask)):
    if outlier_mask[i] == True:
        out_list =  out_list + [df.loc[i,'image_number']]          # rows in which we have outliers
        sum = sum + 1

out_list = set(out_list)        # to get only unique values

### first remove the most extreme outliers
df = df[df['time'] < 3.0]
df = df.reset_index(drop=True)

# drop the outliers, NOT sure we need to do this
#for i in range(0,df.shape[0]):      # for each element in the new dataframe
    # drop the row with that image_number
#    if df.loc[i,'image_number'] in out_list:
        # drop this row
#        df.drop(labels = i, axis = 0, inplace = True)

### standardize for each person
for elem_id in df['id'].unique():
    df.loc[df['id'] == elem_id, 'time'] = stats.zscore(df.loc[df['id'] == elem_id, 'time'], ddof=1)

### convert to a continuous index
## put Non-smiling in the range [-1,0), and the most negative value is the fastest
mask_not_smiling = df.label == 'Not smiling'
df.loc[mask_not_smiling, 'time'] = MinMaxScaler(feature_range=(-1, 0)).fit_transform(df[mask_not_smiling]['time'].values.reshape(-1, 1))

## put Smiling times in the range (0,1], switching the values to get high values for low reaction times
mask_smiling = df.label == 'Smiling'
df.loc[mask_smiling, 'time'] = - df.loc[mask_smiling, 'time']
df.loc[mask_smiling, 'time'] = MinMaxScaler(feature_range=(0, 1)).fit_transform(df[mask_smiling]['time'].values.reshape(-1, 1))

plt.gray()
plt.axis('off')

# def load_images(folder, outliers):
#   images = []
#   files_to_load = os.listdir(folder)
#   for outlier in outliers:
#     files_to_load = files_to_load.remove(outlier)
#   for filename in files_to_load:
#     img = mpimg.imread(os.path.join(folder, filename))
#     if img is not None:
#       img = list(img.reshape(-1, 10, 10))
#       for i in range(len(img)):
#         img[i] = img[i].reshape(-1)
#       images.append(np.array(img))
#   return images

def load_images(folder):
  images = []
  for filename in os.listdir(folder):
    img = mpimg.imread(os.path.join(folder, filename))
    if img is not None:
      img = list(img.reshape(-1, 10, 10))
      for i in range(len(img)):
        img[i] = img[i].reshape(-1)
      images.append(np.array(img))
  return images

images = np.concatenate(load_images("dataset_images"))


# we know we only want components for 95% of variance, so we rewrite the last two lines
pca = PCA() # if we know the number of components we write that, otherwise
                          #   we write the variance we desire (e.g. 0.95)
scores = pca.fit_transform(images)
reconstructed = pca.inverse_transform(scores)
principal_components = pca.components_

# cumulative variance of the components --> 10 give 90% of the variability
var = np.cumsum(pca.explained_variance_ratio_)

threshold = 0.95

plt.figure()
plt.bar(range(0,20), var[:20])
plt.axhline(threshold,color = 'r')
plt.show()

components_to_analyse = 10
learnt_components = list(principal_components[:components_to_analyse, :])

for i in range(components_to_analyse):
    learnt_components[i] = learnt_components[i].reshape(10, 10)

from PIL import Image

for i in range(components_to_analyse):
    img_pc = Image.fromarray(learnt_components[i]*255)
    img_pc.show()

# to save the figure
#plt.savefig("component_" + str(i) + ".png", bbox_inches='tight', pad_inches=0, dpi=226)

# img_compressed = list(pca.inverse_transform(scores))
#
# for i in range(len(img_compressed)):
#   img_compressed[i] = img_compressed[i].reshape(10, 10)
#
# img_compressed = np.array(img_compressed).reshape(1140, -1)
#
# figure = plt.gcf()
# figure.set_size_inches(3.54, 5.04) # per mona lisa ok
# # plt.imshow(img_compressed)
# # plt.savefig('new_img_6.png', bbox_inches='tight', pad_inches=0, dpi=226)