import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

SaveFile_Name = 'all.csv'

file_list = os.listdir('Data/')
# df = pd.read_csv('Data/' + file_list[0])
# df.to_csv('Data/' + SaveFile_Name, encoding="utf_8_sig", index=False)
# for i in range(1, len(file_list)):
#   df = pd.read_csv('Data/' + file_list[i])
#   df.to_csv('Data/' + SaveFile_Name, encoding="utf_8_sig",index=False, header=False, mode='a+')

# count = 0
# for i in range(0, len(file_list)):
#   if file_list[i] == SaveFile_Name:
#     continue
#   if file_list[i] == '.DS_Store':
#     continue
#   count = count + 1
#   ax = plt.subplot(6, 5, count)
#   df = pd.read_csv('Data/' + file_list[i])
#   ax.hist(df['PTS'], bins=10)
#   ax.set_title(file_list[i][:3])
# plt.tight_layout()
# plt.show()

df = pd.read_csv('Data/all.csv')
df = df.drop(columns=['Unnamed: 0'])
# df.hist(column='PTS')
# plt.xlabel("points")
# plt.ylabel("frequency")
# plt.show()

corrmat = df.corr()
# f, ax = plt.subplots(figsize=(20, 18))
# sns.heatmap(corrmat, vmax=.8, square=True)
# plt.show()

k = 12
cols = corrmat.nlargest(k, 'REB')['REB'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()