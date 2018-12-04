import numpy as np
import pandas as pd
from util import TeamsVictoriesPer
from util import PointsDiff

base = 2014
start = 2014
end = 2017

matrix = np.zeros((end - start + 2, 5))

for year in range(start, end + 1):
  path = '/Users/cappuccinuo/Documents/GitHub/Machine-Learning-Stanford/NBA-Prediction/Data/' \
         + str(year) + '-' + str(year + 1)[2:] + '/ScoresRegular' + str(year) + '.txt'
  Scores = np.loadtxt(path, delimiter=' ')

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



m, n = matrix.shape
for i in range (0, n):
  matrix[m - 1][i] = np.mean(matrix[:m - 1, i])
print(matrix)




