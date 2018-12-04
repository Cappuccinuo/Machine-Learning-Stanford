import pandas as pd
import numpy as np
from util import teamToIndex
from os import walk
from os import path
import re

myPath = 'Data/'
for (dirpath, dirnames, filenames) in walk(myPath):
  if (dirpath == myPath):
    continue
  slash = dirpath.rfind('/')
  append = dirpath[slash + 1 : ]
  path = dirpath + '/All_Game_' + append + '.csv'

  df = pd.read_csv(path)
  df['homeIndex'] = df['homeName'].map(lambda x: teamToIndex.get(x))
  df['awayIndex'] = df['awayName'].map(lambda x: teamToIndex.get(x))
  ds = df[['homeIndex', 'homePoints', 'awayIndex', 'awayPoints']]

  horizon = append.rfind('-')
  year = append[:horizon]
  path = dirpath + '/ScoresRegular' + year + '.txt'

  np.savetxt(path, ds.values, fmt='%d')
