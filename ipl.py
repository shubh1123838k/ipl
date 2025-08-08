import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib as plt

matches = pd.read_csv(r"C:\Users\shubh\.cache\kagglehub\datasets\ramjidoolla\ipl-data-set\versions\1\matches.csv")
#,Season,city,date,team1,team2,toss_winner,toss_decision,result,dl_applied,winner,win_by_runs,win_by_wickets,player_of_match,venue,umpire1,umpire2,umpire3
deliveres = pd.read_csv(r"C:\Users\shubh\.cache\kagglehub\datasets\ramjidoolla\ipl-data-set\versions\1\deliveries.csv")
# match_id,inning,batting_team,bowling_team,over,ball,batsman,non_striker,bowler,is_super_over,wide_runs,bye_runs,legbye_runs,noball_runs,penalty_runs,batsman_runs,extra_runs,total_runs,player_dismissed,dismissal_kind,fielder
teams = pd.read_csv(r"C:\Users\shubh\.cache\kagglehub\datasets\ramjidoolla\ipl-data-set\versions\1\teams.csv")
#team1
homewins = pd.read_csv(r"C:\Users\shubh\.cache\kagglehub\datasets\ramjidoolla\ipl-data-set\versions\1\teamwise_home_and_away.csv")
#team,home_wins,away_wins,home_matches,away_matches,home_win_percentage,away_win_percentage
avgruns = pd.read_csv(r"C:\Users\shubh\.cache\kagglehub\datasets\ramjidoolla\ipl-data-set\versions\1\most_runs_average_strikerate.csv")
#batsman,total_runs,out,numberofballs,average,strikerate


# Find the top 5 highest run-scorers across all seasons.
print(deliveres["batsman_runs"].value_counts().head(5))
# Which team has won the most matches overall?
print(matches["winner"].value_counts())
# Count how many matches ended with a tie.
print(matches[matches["winner"] == "tie"].value_counts())
# Find how many matches each team played in total.
team1 = matches['team1'].value_counts()
team2 = matches['team2'].value_counts()
(team1 + team2).sort_values(ascending=False)

# Which player got the most ‘Player of the Match’ awards?
print(matches["player_of_match"].value_counts().idxmax())
# Which stadium hosted the most matches?
print(matches["venue"].value_counts().idxmax())
# Find the strike rate of each batsman (runs per 100 balls).
print(avgruns.groupby("batsman")["strikerate"].value_counts().idxmax())

# Find which team chased down the highest target.
print()

# Calculate the average winning margin (by runs and by wickets) per team.
print() 

# Find the most successful opening pair (1st and 2nd position players together).
print()

# Which team had the best win percentage in the last 3 seasons?
print()

# Which umpire stood in the most number of matches?
print()

# For each player, calculate average runs in powerplay (1–6), middle overs (7–15), and death overs (16–20).
print()

# Build a NumPy array that simulates a rolling average of runs for top 5 batsmen over each match they played.
print()

