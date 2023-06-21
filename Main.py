
!pip install matplotlib==3.4

!pip install pca

# import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import pyplot
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pca
from pca import pca

# read the file
df=pd.read_csv("/content/data.csv")
# Drop first column of dataframe
df = df.iloc[: , 1:]

# display the data
df.head()

# name and types of column
df.info()

"""Top 10 players with most goals"""

dfTop10 = df.nlargest(10, ['goals_x'])

plt.figure(figsize = [20, 10])
bar1 = plt.bar(dfTop10['player_name'],dfTop10['goals_x'])
plt.bar_label(bar1, padding=3, fontsize=12)
plt.title("Player with most number of Goals",fontsize = 15)
plt.ylabel("Total Number")
plt.savefig("1")

plt.show()

"""Top 10 players with most number of matches played"""

dftop10 = df.nlargest(10, ['match_played_x'])

plt.figure(figsize = [20, 10])
bar1 = plt.bar(dftop10['player_name'],dftop10['match_played_x'])
plt.bar_label(bar1, padding=3, fontsize=12)
plt.title("Players with most number of matches played",fontsize = 15)
plt.ylabel("Total Number")
plt.savefig("2")
plt.show()

"""Top 10 Goal keepers with most saved goals """

dfTop10 = df.nlargest(10, ['saved'])

plt.figure(figsize = [20, 10])
bar1 = plt.bar(dfTop10['player_name'],dfTop10['saved'])
plt.bar_label(bar1, padding=3, fontsize=12)
plt.title("Goal Keeper with most saved number of Goals",fontsize = 15)
plt.ylabel("Total Number")
plt.savefig("3")

plt.show()

"""Top 5 Goal keepers with most saved Penalties"""

dfTop10 = df.nlargest(5, ['saved_penalties'])

plt.figure(figsize = [20, 10])
bar1 = plt.bar(dfTop10['player_name'],dfTop10['saved_penalties'])
plt.bar_label(bar1, padding=3, fontsize=12)
plt.title("Goal Keeper with most saved number of Penalties",fontsize = 15)
plt.ylabel("Total Number")
plt.savefig("4")

plt.show()

"""Top 10 defenders"""

dfTop10 = df.nlargest(10, ['blocked'])

plt.figure(figsize = [20, 10])
bar1 = plt.bar(dfTop10['player_name'],dfTop10['blocked'])
plt.bar_label(bar1, padding=3, fontsize=12)
plt.title("Player with most defends",fontsize = 15)
plt.ylabel("Total Number")
plt.savefig("5")

plt.show()

"""Top 10 players with most numbers of minutes played"""

dfTop10 = df.nlargest(10, ['minutes_played_x'])

plt.figure(figsize = [20, 10])
bar1 = plt.bar(dfTop10['player_name'],dfTop10['minutes_played_x'])
plt.bar_label(bar1, padding=3, fontsize=12)
plt.title("Player with most defends",fontsize = 15)
plt.ylabel("Total Number of Minutes")
plt.savefig("6")

plt.show()

"""Top 10 Mid-Fileders"""

dfTop10 = df.nlargest(10, ['balls_recoverd_x'])

plt.figure(figsize = [20, 10])
bar1 = plt.bar(dfTop10['player_name'],dfTop10['balls_recoverd_x'])
plt.bar_label(bar1, padding=3, fontsize=12)
plt.title("Top 10 Mid-Fileders",fontsize = 15)
plt.ylabel("Total Number of Balls Recovered")
plt.savefig("7")

plt.show()

"""Top 10 palyers with most number of Free Kicks"""

dfTop10 = df.nlargest(10, ['freekicks_taken'])

plt.figure(figsize = [20, 10])
bar1 = plt.bar(dfTop10['player_name'],dfTop10['freekicks_taken'])
plt.bar_label(bar1, padding=3, fontsize=12)
plt.title("Top 10 palyers with most number of free kicks",fontsize = 15)
plt.ylabel("Total Number of Free Kicks")
plt.savefig("8")

plt.show()

"""Top 10 Goal Keepers with most number of Goal Conceded"""

dfTop10 = df.nlargest(10, ['conceded'])

plt.figure(figsize = [20, 10])
bar1 = plt.bar(dfTop10['player_name'],dfTop10['conceded'])
plt.bar_label(bar1, padding=3, fontsize=12)
plt.title("Top 10 Goal Keepers with most number of Goal Conceded",fontsize = 15)
plt.ylabel("Total Number of Goals")
plt.savefig("9")
plt.show()

"""PCA"""

df['position'].unique()

"""Goal Keeper"""

df2=df.query("position == 'Goalkeeper'")

df2=df2[['saved_penalties','saved','conceded','balls_recoverd_x','minutes_played_x']]

# Initialize
model = pca(n_components=2, normalize=True)
# Fit transform
out = model.fit_transform(df2)

out['topfeat']

# Make plot with only the directions (no scatter)
fig, ax = model.biplot(label=True, legend=False, figsize=(10, 6))

"""Forward"""

df2=df.query("position == 'Forward'")

df2=df2[['goals_x','assists_x','balls_recoverd_x','minutes_played_x','match_played_x']]

# Initialize
model = pca(n_components=2, normalize=True)
# Fit transform
out = model.fit_transform(df2)

out['topfeat']

# Make plot with only the directions (no scatter)
fig, ax = model.biplot(label=True, legend=False, figsize=(10, 6))

"""Midfielder"""

df2=df.query("position == 'Midfielder'")

df2=df2[['goals_x','assists_x','balls_recoverd_x','minutes_played_x','blocked','match_played_x']]

# Initialize
model = pca(n_components=2, normalize=True)
# Fit transform
out = model.fit_transform(df2)

out['topfeat']

# Make plot with only the directions (no scatter)
fig, ax = model.biplot(label=True, legend=False, figsize=(10, 6))

"""Defender"""

df2=df.query("position == 'Defender'")

df2=df2[['balls_recoverd_x','minutes_played_x','blocked','saved_penalties','match_played_x']]

# Initialize
model = pca(n_components=2, normalize=True)
# Fit transform
out = model.fit_transform(df2)

out['topfeat']

# Make plot with only the directions (no scatter)
fig, ax = model.biplot(label=True, legend=False, figsize=(10, 6))
