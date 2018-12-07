'''
1st step
Get Scores Regular year txt
Format:
homeIndex homeScore awayIndex awayScore
'''

import pandas as pd
import numpy as np
from util import teamToIndex
from os import walk
from os import path

myPath = './Data/'
for (dirpath, dirnames, filenames) in walk(myPath):
  if (dirpath == myPath):
    continue
  slash = dirpath.rfind('/')
  append = dirpath[slash + 1 : ]
  p = dirpath + '/All_Game_in_time_order' + append + '.csv'
  if (not path.exists(p)):
    continue
  df = pd.read_csv(p)
  df['homeIndex'] = df['homeName'].map(lambda x: teamToIndex.get(x))
  df['awayIndex'] = df['awayName'].map(lambda x: teamToIndex.get(x))
  ds = df[['homeIndex', 'homePoints', 'awayIndex', 'awayPoints']]

  horizon = append.rfind('-')
  year = append[:horizon]
  p = myPath + 'RegularScoreTxt' + '/ScoresRegular' + year + '.txt'

  np.savetxt(p, ds.values, fmt='%d')
