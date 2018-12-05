import numpy as np
import pandas as pd
from util import TeamsVictoriesPer
from util import PointsDiff
from util import LastNGamesPer
from util import VisitorVictoriesPer
from util import HomeVictoriesPer
from util import HistoryTeams

base = 2014
start = 2014
end = 2017

matrix = np.zeros((end - start + 2, 5))
FeatureVector = np.ndarray(shape=(1, 4))

for year in range(start, end + 1):
  path = '/Users/cappuccinuo/Documents/GitHub/Machine-Learning-Stanford/NBA-Prediction/Data/' \
         + str(year) + '-' + str(year + 1)[2:] + '/ScoresRegular' + str(year) + '.txt'
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

  df1 = pd.DataFrame(TVP)
  df2 = pd.DataFrame(LNGP)
  df3 = pd.DataFrame(HVP)
  df4 = pd.DataFrame(VVP)
  df5 = pd.DataFrame(PD)
  frames = [df, df1, df2, df3, df4, df5]
  df = pd.concat(frames, axis=1)
  df.columns = ['homeIndex', 'homeScore', 'awayIndex', 'awayScore', 'HTVP', 'ATVP', 'HLNGP', 'ALNGP', 'HVP', 'VVP', 'HPD', 'APD']
  df['rst'] = (df['homeScore'] > df['awayScore']) * 1
  fn = str(year) + '-' + str(year + 1)[2:] + '_features.csv'
  df.to_csv(fn)

# m, n = matrix.shape
# for i in range (0, n):
#   matrix[m - 1][i] = np.mean(matrix[:m - 1, i])
# print(matrix)
# matrix = [[0.6495935, 0.65243902, 0.59593496, 0.62926829, 0.58773181],
#  [0.65650407, 0.67073171, 0.61422764, 0.65081301, 0.61951909],
#  [0.59796748, 0.60650407, 0.56626016, 0.5995935, 0.55827338],
#  [0.62317073, 0.64268293, 0.58211382, 0.62479675, 0.58981612],
#  [0.63180894, 0.64308943, 0.58963415, 0.62611789,0.5888351]]
# plotDataToTable(matrix)






