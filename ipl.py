import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import streamlit as st

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


# Top 5 highest run-scorers across all seasons
top_batsmen = deliveres.groupby("batsman")['batsman_runs'].sum().sort_values(ascending=False).head(5)
print(top_batsmen)

# Team with most wins overall
most_wins = matches['winner'].value_counts().idxmax()
print(f"Most winning team: {most_wins}")

# Matches ended with a tie
tie_matches = matches[matches['result'] == 'tie']
print(f"Number of tied matches: {len(tie_matches)}")

# Matches played by each team
team1 = matches['team1'].value_counts()
team2 = matches['team2'].value_counts()
total_matches = (team1 + team2).sort_values(ascending=False)
print(total_matches)

# Player with most 'Player of the Match' awards
pom_player = matches['player_of_match'].value_counts().idxmax()
print(f"Most awards: {pom_player}")

# Stadium hosting most matches
most_venue = matches['venue'].value_counts().idxmax()
print(f"Most matches hosted at: {most_venue}")

# Strike rate of each batsman
top_strikerate = avgruns.sort_values('strikerate', ascending=False).head(5)[['batsman', 'strikerate']]
print(top_strikerate)

# Team chased down highest target
chased = matches[matches['result'] == 'normal']
highest_target = deliveres.groupby('match_id')['total_runs'].sum().max()
highest_target_match = deliveres.groupby('match_id')['total_runs'].sum().idxmax()
chasing_team = matches.loc[matches['id'] == highest_target_match, 'winner'].values[0]
print(f"Team: {chasing_team}, Target: {highest_target}")

# Average winning margin per team
margin = matches.groupby('winner')[['win_by_runs', 'win_by_wickets']].mean()
print(margin)

# Most successful opening pair
openings = deliveres[deliveres['over'] == 1].groupby(['match_id', 'batting_team'])['batsman'].unique()
pairs = openings.apply(lambda x: tuple(x[:2]) if len(x) >= 2 else None)
pair_counts = pairs.value_counts().head(1)
print(pair_counts)

# Best win percentage in last 3 seasons
last_seasons = matches['Season'].sort_values(ascending=False).unique()[:3]
recent = matches[matches['Season'].isin(last_seasons)]
win_pct = recent['winner'].value_counts() / (recent['team1'].value_counts() + recent['team2'].value_counts())
print(win_pct.sort_values(ascending=False).head(1))

# Umpire in most matches
umps = pd.concat([matches['umpire1'], matches['umpire2']]).value_counts().idxmax()
print(f"Umpire: {umps}")

# Average runs in powerplay, middle, death overs per player
def phase_avg(df, start, end):
    phase = df[(df['over'] >= start) & (df['over'] <= end)]
    return phase.groupby('batsman')['batsman_runs'].mean().sort_values(ascending=False).head(5)
print("Powerplay (1-6):")
print(phase_avg(deliveres, 1, 6))
print("Middle (7-15):")
print(phase_avg(deliveres, 7, 15))
print("Death (16-20):")
print(phase_avg(deliveres, 16, 20))

# Rolling average of runs for top 5 batsmen
top5 = deliveres.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False).head(5).index
for batsman in top5:
    player_df = deliveres[deliveres['batsman'] == batsman].sort_values('match_id')
    rolling = player_df['batsman_runs'].rolling(window=5, min_periods=1).mean()
    print(f"Rolling average for {batsman}:")
    print(rolling)

# Visualizations

# Top 5 highest run-scorers barplot
plt.figure(figsize=(8,4))
sb.barplot(x=top_batsmen.index, y=top_batsmen.values)
plt.title('Top 5 Highest Run-Scorers')
plt.xlabel('Batsman')
plt.ylabel('Runs')
plt.tight_layout()
plt.show()

# Team with most wins overall barplot
plt.figure(figsize=(10,4))
team_wins = matches['winner'].value_counts()
sb.barplot(x=team_wins.index, y=team_wins.values)
plt.title('Total Wins by Team')
plt.xlabel('Team')
plt.ylabel('Wins')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Stadium hosting most matches barplot
plt.figure(figsize=(10,4))
venue_counts = matches['venue'].value_counts().head(10)
sb.barplot(x=venue_counts.index, y=venue_counts.values)
plt.title('Top 10 Stadiums by Matches Hosted')
plt.xlabel('Venue')
plt.ylabel('Matches Hosted')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Strike rate of top 5 batsmen
plt.figure(figsize=(8,4))
sb.barplot(x=top_strikerate['batsman'], y=top_strikerate['strikerate'])
plt.title('Top 5 Batsmen by Strike Rate')
plt.xlabel('Batsman')
plt.ylabel('Strike Rate')
plt.tight_layout()
plt.show()

# Rolling average of runs for top 5 batsmen (line plot)
for batsman in top5:
    player_df = deliveres[deliveres['batsman'] == batsman].sort_values('match_id')
    rolling = player_df['batsman_runs'].rolling(window=5, min_periods=1).mean()
    plt.plot(rolling.values, label=batsman)
plt.title('Rolling Average of Runs (Top 5 Batsmen)')
plt.xlabel('Innings')
plt.ylabel('Rolling Avg Runs')
plt.legend()
plt.tight_layout()
plt.show()

