import numpy as np
import pandas as pd

base = 2014
start = 2014
end = 2017
matrix = np.zeros((end - start + 1, 82))
for year in range(start, end + 1):
  print(year)
  path = '/Users/cappuccinuo/Documents/GitHub/Machine-Learning-Stanford/NBA-Prediction/Data/' \
         + str(year) + '-' + str(year + 1)[2:] + '/ScoresRegular' + str(year) + '.txt'
  Scores = np.loadtxt(path, delimiter=' ')

  # two tables, one with the percentage of Victories until the i game of the team
  # the other with the number of wins until the i game of the team
  percentageVictory = np.zeros((82, 30))
  numberOfVictories = np.zeros((82, 30))

  for team in range(1, 31):
    df = pd.DataFrame(Scores)
    teamGames = df.index[(df[0] == team) | (df[2] == team)].tolist()
    teamGames.sort()
    wins = 0

    for i in range(0, len(teamGames)):
      game = teamGames[i]
      homeTeam = Scores[game][0]
      homeScore = Scores[game][1]
      awayTeam = Scores[game][2]
      awayScore = Scores[game][3]
      if (homeTeam == team and homeScore > awayScore):
        wins += 1
      elif (awayTeam == team and awayScore > homeScore):
        wins += 1
      percentageVictory[i][team - 1] = wins / (i + 1)
      numberOfVictories[i][team - 1] = wins

  for game in range(0, 82):
    firstCount = game * 15
    correctPred = 0
    impredictable = 0
    for i in range(firstCount, len(Scores)):
      df = pd.DataFrame(Scores)
      t1 = int(df.iloc[i][0])
      t2 = int(df.iloc[i][2])
      ds = df.iloc[0:i]
      team1Games = len(ds.index[(ds[0] == t1) | (ds[2] == t1)].tolist())
      team2Games = len(ds.index[(ds[0] == t2) | (ds[2] == t2)].tolist())
      s1 = df.iloc[i][1]
      s2 = df.iloc[i][3]

      if (team1Games - 2 > 0 and team2Games - 2 > 0):
        homeWin = (percentageVictory[team1Games - 2][t1 - 1] > percentageVictory[team2Games - 2][t2 - 1]) and s1 > s2
        awayWin = (percentageVictory[team2Games - 2][t2 - 1] > percentageVictory[team1Games - 2][t1 - 1]) and s2 > s1
        sameRate = percentageVictory[team1Games - 2][t1 - 1] == percentageVictory[team2Games - 2][t2 - 1]
        if (homeWin or awayWin):
          correctPred += 1
        elif (sameRate):
          impredictable += 1
      else:
        impredictable += 1
    matrix[year - base][game] = (correctPred + (impredictable / 2)) / (len(Scores) - firstCount)
print(matrix)