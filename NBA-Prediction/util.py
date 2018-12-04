import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def PointsDiff(ScoresData):
    result = np.zeros((len(ScoresData), 2))
    result[:, :] = -9999

    for team in range(1, 31):
        df = pd.DataFrame(ScoresData)
        teamGames = df.index[(df[0] == team) | (df[2] == team)].tolist()
        teamGames.sort()
        teamPD = 0

        for i in range(1, len(teamGames)):
            prevGame = teamGames[i - 1]
            homeTeam = ScoresData[prevGame][0]
            homeScore = ScoresData[prevGame][1]
            awayTeam = ScoresData[prevGame][2]
            awayScore = ScoresData[prevGame][3]
            if (homeTeam == team):
                teamPD += homeScore - awayScore;
            elif (awayTeam == team):
                teamPD += awayScore - homeScore;
            currGame = teamGames[i]
            currHomeTeam = ScoresData[currGame][0]
            if (currHomeTeam == team):
                result[currGame][0] = teamPD / i
            else:
                result[currGame][1] = teamPD / i
    return result

def LastNGamesPer(ScoresData, N):
    result = np.zeros((len(ScoresData), 2))
    result[:, :] = -9999

    for team in range(1, 31):
        df = pd.DataFrame(ScoresData)
        teamGames = df.index[(df[0] == team) | (df[2] == team)].tolist()
        teamGames.sort()
        wins = np.zeros((N, 1))

        for i in range(0, N):
            game = teamGames[i]
            homeTeam = ScoresData[game][0]
            homeScore = ScoresData[game][1]
            awayTeam = ScoresData[game][2]
            awayScore = ScoresData[game][3]

            if (homeTeam == team and homeScore > awayScore) :
                wins[i] = 1
            elif (awayTeam == team and awayScore > homeScore):
                wins[i] = 1

        k = 0
        for i in range(N, len(teamGames)):
            game = teamGames[i]
            homeTeam = ScoresData[game][0]
            homeScore = ScoresData[game][1]
            awayTeam = ScoresData[game][2]
            awayScore = ScoresData[game][3]
            if (homeTeam == team):
                result[game][0] = np.mean(wins)
            else:
                result[game][1] = np.mean(wins)

            if (homeTeam == team and homeScore > awayScore):
                wins[k] = 1
            elif (awayTeam == team and awayScore > homeScore):
                wins[k] = 1
            else :
                wins[k] = 0
            k += 1
            if (k == N):
                k = 0
    return result

def HomeVictoriesPer(ScoresData):
    result = np.zeros((len(ScoresData), 1))
    for i in range(0, len(ScoresData)):
        result[i] = -9999

    for team in range(1, 31):
        df = pd.DataFrame(ScoresData)
        teamGames = df.index[(df[0] == team)].tolist()
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

            currGame = teamGames[i]
            result[currGame] = wins / i
    return result

def VisitorVictoriesPer(ScoresData):
    result = np.zeros((len(ScoresData), 1))
    for i in range(0, len(ScoresData)):
        result[i] = -9999

    for team in range(1, 31):
        df = pd.DataFrame(ScoresData)
        teamGames = df.index[(df[2] == team)].tolist()
        teamGames.sort()
        wins = 0

        for i in range(1, len(teamGames)):
            prevGame = teamGames[i - 1]
            homeTeam = ScoresData[prevGame][0]
            homeScore = ScoresData[prevGame][1]
            awayTeam = ScoresData[prevGame][2]
            awayScore = ScoresData[prevGame][3]
            if (awayTeam == team and awayScore > homeScore):
                wins += 1

            currGame = teamGames[i]
            result[currGame] = wins / i
    return result

def HistoryTeams(ScoresData):
    result = np.zeros((30, 30))
    HT = np.zeros((len(ScoresData), 1))
    for i in range (0, len(ScoresData)):
        HT[i] = -9999
    for t1 in range(1, 31):
        for t2 in range(t1 + 1, 31):
            team1 = t1
            team2 = t2
            df = pd.DataFrame(ScoresData)
            teamGames = df.index[(((df[0] == team1) & (df[2] == team2))
                                 | ((df[0] == team2) & (df[2] == team1)))].tolist()
            teamGames.sort()
            winsTeam1 = 0

            for i in range(1, len(teamGames)):
                prevGame = teamGames[i - 1]
                homeTeam = ScoresData[prevGame][0]
                homeScore = ScoresData[prevGame][1]
                awayTeam = ScoresData[prevGame][2]
                awayScore = ScoresData[prevGame][3]
                if (homeTeam == team1):
                    if (homeScore > awayScore):
                        winsTeam1 += 1
                elif (awayTeam == team1):
                    if (awayScore > homeScore):
                        winsTeam1 += 1
                currGame = teamGames[i]
                currHomeTeam = ScoresData[currGame][0]
                currAwayTeam = ScoresData[currGame][2]
                if (currHomeTeam == team1):
                    HT[currGame] = winsTeam1 - (i - winsTeam1)
                elif (currAwayTeam == team1):
                    HT[currGame] = -(winsTeam1 - (i - winsTeam1))
                result[team1 - 1][team2 - 1] = (winsTeam1 - (i - winsTeam1))
    for i in range(0, 30):
        for j in range(0, 30):
            result[j][i] = -result[i][j]
    return HT, result

def plotDataToTable(matrix):
    columns = ('A', 'B', 'C', 'D', 'E')
    rows = ['2014-15', '2015-16', '2016-17', '2017-18', 'mean']
    values = np.arange(0, 1, 0.1)
    value_increment = 0.1

    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    n_rows = len(matrix)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        plt.bar(index, matrix[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + matrix[row]
        cell_text.append(['%.1f' % (x) for x in y_offset])
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    cell_text.reverse()

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          loc='bottom')

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.ylabel("Prediction Accuracy".format(value_increment))
    plt.yticks(values * value_increment, ['%d' % val for val in values])
    plt.xticks([])
    plt.title('Data Analysis')

    plt.show()