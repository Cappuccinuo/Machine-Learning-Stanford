import numpy as np
import pandas as pd
teamToIndex = {
    'ATL': 1,
    'BOS': 2,
    'BKN': 3,
    'CHA': 4,
    'CHI': 5,
    'CLE': 6,
    'DAL': 7,
    'DEN': 8,
    'DET': 9,
    'GSW': 10,
    'HOU': 11,
    'IND': 12,
    'LAC': 13,
    'LAL': 14,
    'MEM': 15,
    'MIA': 16,
    'MIL': 17,
    'MIN': 18,
    'NOP': 19,
    'NYK': 20,
    'OKC': 21,
    'ORL': 22,
    'PHI': 23,
    'PHX': 24,
    'POR': 25,
    'SAC': 26,
    'SAS': 27,
    'TOR': 28,
    'UTA': 29,
    'WAS': 30,
}

def TeamsVictoriesPer(ScoresData):
  result = np.zeros((len(ScoresData), 2))
  result[:,:] = -9999

  for team in range(1, 31):
    df = pd.DataFrame(ScoresData)
    teamGames = df.index[(df[0] == team) | (df[2] == team)].tolist()
    teamGames.sort()
    wins = 0

    for i in range(1, len(teamGames)):
        prevGame = teamGames[i - 1]
        homeTeam = ScoresData[prevGame][0]
        homeScore = ScoresData[prevGame][1]
        awayTeam = ScoresData[prevGame][2]
        awayScore = ScoresData[prevGame][3]
        if (homeTeam == team and homeScore > awayScore):
            wins += 1
        elif (awayTeam == team and awayScore > homeScore):
            wins += 1

        currGame = teamGames[i]
        currHomeTeam = ScoresData[currGame][0]
        if (currHomeTeam == team):
            result[currGame][0] = wins / i
        else :
            result[currGame][1] = wins / i
  return result