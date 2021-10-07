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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

filename = 'data.csv'
df = pd.read_csv(filename, sep=';')  # gives a dataframe

### first remove the most extreme outliers
df = df[df['time'] < 4.0]
df = df.reset_index(drop=True)

### standardize for each person
for elem_id in df['id'].unique():
  df.loc[df['id'] == elem_id, 'time'] = stats.zscore(df.loc[df['id'] == elem_id, 'time'], ddof=1)

# remove outliers after the standardization
std = np.std(df['time'],ddof= 1)        # almost 1, since we have standardized, but there's always an error

df = df[df['time'] < std*2]
df = df[df['time'] > -2*std]
df = df.reset_index(drop=True)

# %% convert to a continuous index

## put Non-smiling in the range [-1,0), and the most negative value is the fastest
mask_not_smiling = df.label == 'Not smiling'
df.loc[mask_not_smiling, 'time'] = MinMaxScaler(feature_range=(-1, 0)).fit_transform(df[mask_not_smiling]['time'].values.reshape(-1, 1))

## put Smiling times in the range (0,1], switching the values to get high values for low reaction times
mask_smiling = df.label == 'Smiling'
df.loc[mask_smiling, 'time'] = - df.loc[mask_smiling, 'time']
df.loc[mask_smiling, 'time'] = MinMaxScaler(feature_range=(0, 1)).fit_transform(df[mask_smiling]['time'].values.reshape(-1, 1))


# %% PCA ON THE IMAGES

plt.gray()

def load_images(folder):
  images = []
  for filename in sorted(os.listdir(folder)):
    try:
      img = mpimg.imread(os.path.join(folder, filename))
      if img is not None:
        images.append(img.reshape(-1))
    except:
      continue
  return images

images = np.stack(load_images("dataset_images"))

# we know we only want components for _% of variance, so we rewrite the last two lines
pca = PCA(n_components=0.90) # if we know the number of components we write that, otherwise
                            #   we write the variance we desire (e.g. 0.95)
scores = pca.fit_transform(images)
# for i in range(len(pca.components_)):
#   scores[:, i] = scores[:, i] + pca.mean_[i]

# variance explained
var = np.cumsum(pca.explained_variance_ratio_)

###### PER VELOCIZZARE, DA UNCOMMENT DOPO (QUESTO FA SOLO RICOSTRUZIONE DELLE IMMAGINI PER SALVARLE
# principal_components = pca.components_
# components_to_analyse = len(principal_components)
# learnt_components = list(principal_components[:components_to_analyse, :])
# for i in range(len(learnt_components)):
#   learnt_components[i] = (learnt_components[i] + pca.mean_).reshape(50,-1)  #add mean
#
# figure = plt.gcf()
# # figure.set_size_inches(0.04444, 0.04444)
# for i in range(components_to_analyse):
#   plt.imshow(learnt_components[i])
#   plt.savefig("component_"+str(i+1)+".png", bbox_inches='tight', pad_inches=0, dpi=226)
#
# reconstructed = list(pca.inverse_transform(scores))
#
# rows_per_image = len(reconstructed)//len(df['image_number'].unique())
# for i in range(len(df['image_number'].unique())):
#   reconstructed[i] = reconstructed[i:(i+rows_per_image)]
#   reconstructed[i+1:len(reconstructed)] = reconstructed[(i+rows_per_image):len(reconstructed)]
#
# for i in range(len(reconstructed)):
#   reconstructed[i] = np.array(reconstructed[i]).reshape(50, 50)
# for i in range(len(reconstructed)):
#   plt.imshow(reconstructed[i])
#   plt.savefig("reconstructed_"+str(i+1)+".png", bbox_inches='tight', pad_inches=0, dpi=226)
###### PER VELOCIZZARE, DA UNCOMMENT DOPO FINO A QUI

df = df.sort_values(by=['image_number', 'id'])
df = df.reset_index(drop=True)

df_image_numbers = df.image_number
df_times = df.time

### we ridivide the scores in ech image
# scores_x = list(scores)
# rows_per_image = len(scores_x)//len(df_image_numbers.unique())
# for i in range(len(df_image_numbers.unique())):
#   scores_x[i] = scores_x[i:(i+rows_per_image)]
#   scores_x[i+1:len(scores_x)] = scores_x[(i+rows_per_image):len(scores_x)]
# for i in range(len(scores_x)):
#   scores_x[i] = np.concatenate(scores_x[i])
### not needed anymore

images_for_regression = []
for i in range(len(df_image_numbers.unique())):
  for _ in range(len(df_image_numbers[df_image_numbers == (i+1)])):
    images_for_regression.append(scores_x[i])

# Model initialization
regression_model = LinearRegression()
# Fit the data(train the model)
regression_model.fit(images_for_regression, df_times)
# Predict
df_times_predicted = regression_model.predict(images_for_regression)

# model evaluation
rmse = mean_squared_error(df_times, df_times_predicted)
r2 = r2_score(df_times, df_times_predicted)

# printing values
print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)

coeff = regression_model.coef_
beta0 = regression_model.intercept_

num = []
for i in range(len(df_image_numbers.unique())):  # for each image
  # take the number of times the image is repeated
  num += [len(df_image_numbers[df_image_numbers == (i + 1)])]

new_times = []
n = 0
# now take the first n elements of the vector df_times_predicted and substitute with one
for i in range(len(df_image_numbers.unique())):
  new_times += [df_times_predicted[n]]
  n = n + num[i]

new_times = np.array(new_times)
plt.scatter(range(len(scores_x)), new_times)
plt.plot()
plt.show()

# %%

nn2 = np.linalg.norm(coeff)**2

idx = +1.5       # a person which is pretty clear is smiling

alpha = (idx - beta0)/nn2

new_img = alpha*coeff          # these are the pixels of the new image

new_img_reconstructed = pca.inverse_transform(new_img.T)
new_img_reconstructed = new_img_reconstructed.reshape(50,-1)
plt.imshow(new_img_reconstructed)
plt.savefig('smiling_2.png')
