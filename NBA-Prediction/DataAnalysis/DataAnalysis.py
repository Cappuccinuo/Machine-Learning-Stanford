'''
2nd step
'''

import numpy as np
import pandas as pd
from util import TeamsVictoriesPer
from util import PointsDiff
from util import LastNGamesPer
from util import VisitorVictoriesPer
from util import HomeVictoriesPer
from util import HistoryTeams

def process_row(row):
  a = row['homeScore']
  b = row['awayScore']
  data =[1] if a > b else [-1]
  return pd.Series(np.array(data, dtype=int))

base = 2010
start = 2010
end = 2017

matrix = np.zeros((end - start + 2, 5))
FeatureVector = np.ndarray(shape=(1, 4))

for year in range(start, end + 1):
  path = '../Data/RegularScoreTxt/ScoresRegular' + str(year) + '.txt'
  Scores = np.loadtxt(path, delimiter=' ')
  df = pd.DataFrame(Scores)

  # Prediction based on Teams Victories Percentage before the predicted game
  TVP = TeamsVictoriesPer(Scores)
  correctPred = 0
  impredictable = 0
  for i in range(0, len(Scores)):
    if (TVP[i][0] == -9999 or TVP[i][1] == -9999):
      impredictable += 1
    else:
      homeWinCorrect = (TVP[i][0] > TVP[i][1]) and (Scores[i][1] > Scores[i][3])
      awayWinCorrect = (TVP[i][1] > TVP[i][0]) and (Scores[i][3] > Scores[i][1])
      sameRate = (TVP[i][0] == TVP[i][1])

      if (homeWinCorrect or awayWinCorrect):
        correctPred += 1
      elif (sameRate):
        impredictable += 1
  matrix[year - base][0] = (correctPred + impredictable / 2) / len(Scores)

  # Prediction based on Teams point differential per game before the predicted game
  PD = PointsDiff(Scores)
  correctPred = 0
  impredictable = 0
  for i in range(0, len(Scores)):
    if (PD[i][0] == -9999 or PD[i][1] == -9999):
      impredictable += 1
    else:
      homeWinCorrect = (PD[i][0] > PD[i][1]) and (Scores[i][1] > Scores[i][3])
      awayWinCorrect = (PD[i][1] > PD[i][0]) and (Scores[i][3] > Scores[i][1])
      sameRate = (PD[i][0] == PD[i][1])

      if (homeWinCorrect or awayWinCorrect):
        correctPred += 1
      elif (sameRate):
        impredictable += 1
  matrix[year - base][1] = (correctPred + impredictable / 2) / len(Scores)

  # Prediction based on win-loss percentage in the previous N games of both teams
  # before the predicted game
  N = 8
  LNGP = LastNGamesPer(Scores, N)
  correctPred = 0
  impredictable = 0
  for i in range(0, len(Scores)):
    if (LNGP[i][0] == -9999 or LNGP[i][1] == -9999):
      impredictable += 1
    else:
      homeWinCorrect = (LNGP[i][0] > LNGP[i][1]) and (Scores[i][1] > Scores[i][3])
      awayWinCorrect = (LNGP[i][1] > LNGP[i][0]) and (Scores[i][3] > Scores[i][1])
      sameRate = (LNGP[i][0] == LNGP[i][1])

      if (homeWinCorrect or awayWinCorrect):
        correctPred += 1
      elif (sameRate):
        impredictable += 1
  matrix[year - base][2] = (correctPred + impredictable / 2) / len(Scores)


  # Prediction based on visitor team win-loss percentage as visitor
  # and home team win-loss percentage as home
  # before the predicted game

  HVP = HomeVictoriesPer(Scores)
  VVP = VisitorVictoriesPer(Scores)
  correctPred = 0
  impredictable = 0
  for i in range(0, len(Scores)):
    if (HVP[i] == -9999 or VVP[i] == -9999):
      impredictable += 1
    else:
      homeWinCorrect = (HVP[i] > VVP[i]) and (Scores[i][1] > Scores[i][3])
      awayWinCorrect = (VVP[i] > HVP[i]) and (Scores[i][3] > Scores[i][1])
      sameRate = (HVP[i] == VVP[i])

      if (homeWinCorrect or awayWinCorrect):
        correctPred += 1
      elif (sameRate):
        impredictable += 1
  matrix[year - base][3] = (correctPred + impredictable / 2) / len(Scores)

  # Predictions based on the results of previous games between the teams
  HT, SeasonStanding = HistoryTeams(Scores)
  correctPred = 0
  impredictable = 0
  for i in range(0, len(Scores)):
    if (HT[i] == -9999):
      impredictable += 1
    else:
      homeWinCorrect = (HT[i] > 0) and (Scores[i][1] > Scores[i][3])
      awayWinCorrect = (HT[i] < 0) and (Scores[i][3] > Scores[i][1])
      sameRate = (HT[i] == 0)

      if (homeWinCorrect or awayWinCorrect):
        correctPred += 1
      elif (sameRate):
        impredictable += 1

  matrix[year - base][4] = (correctPred) / (len(Scores) - impredictable)

  df1 = pd.DataFrame(HVP)
  df2 = pd.DataFrame(VVP)
  df3 = pd.DataFrame(TVP)
  df4 = pd.DataFrame(PD)
  df5 = pd.DataFrame(LNGP)
  frames = [df, df1, df2, df3, df4, df5]
  df = pd.concat(frames, axis=1)
  df.columns = ['homeIndex', 'homeScore', 'awayIndex', 'awayScore', 'HVP', 'VVP', 'HTVP', 'ATVP', 'HPD', 'APD', 'HLNGP', 'ALNGP']
  ds = df.apply(process_row, axis = 1)
  ds.columns = ['rst']
  df = df.join(ds)
  fn = str(year) + '-' + str(year + 1)[2:] + '_features.csv'
  df.to_csv('../Data/SeperateFeatures/' + fn, index=False)

m, n = matrix.shape
for i in range (0, n):
  matrix[m - 1][i] = np.mean(matrix[:m - 1, i])
print(matrix)
# [[0.65243902 0.65813008 0.61463415 0.65853659 0.58123249]
#  [0.63080808 0.62626263 0.58484848 0.64343434 0.60554371]
#  [0.64593496 0.65894309 0.61544715 0.64390244 0.59123055]
#  [0.65243902 0.64227642 0.61585366 0.64186992 0.56795422]
#  [0.66788618 0.67113821 0.61300813 0.64878049 0.58773181]
#  [0.65813008 0.65691057 0.61300813 0.66178862 0.61951909]
#  [0.60650407 0.62195122 0.59186992 0.61869919 0.55827338]
#  [0.63495935 0.62845528 0.61219512 0.63943089 0.58981612]
#  [0.6436376  0.64550844 0.60760809 0.64455531 0.58766267]]






