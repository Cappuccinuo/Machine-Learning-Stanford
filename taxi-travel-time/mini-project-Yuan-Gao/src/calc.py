import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import time

#df = pd.read_csv('data/train.csv', converters={'POLYLINE': lambda x: json.loads(x)})
df = pd.read_csv('../data/train_new_V5.csv')
df.drop(['year'], axis=1, inplace=True)
y = df['len']*15 + 1

# sns.set_style('white')
# sns.set_context("paper",font_scale=2)
# corr = df.corr()
# mask = np.zeros_like(corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# f, ax = plt.subplots(figsize=(11,9))
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0,
#            square=True, linewidths=0.5, cbar_kws={"shrink":0.5})
# plt.show()
#
# fig, ax = plt.subplots(ncols=1, nrows=1)
# ax.set_xlim(0, 30)
# sns.distplot(y[pd.isnull(y) == False]/3600,ax=ax,bins=100,kde=False,hist_kws={'log':True})
# plt.show()
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12,10))
plt.ylim(40.8, 41.5)
plt.xlim(-8.2, -8.8)
ax.scatter(df['xs'],df['ys'], s=0.01, alpha=1)
plt.show()