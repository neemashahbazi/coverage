import pandas as pd

df = pd.read_csv("data/player_regular_season-Trimmed.csv")
df.drop("ilkid", axis=1, inplace=True)
df.drop("year", axis=1, inplace=True)
df.drop("firstname", axis=1, inplace=True)
df.drop("lastname", axis=1, inplace=True)
df.drop("team", axis=1, inplace=True)
df.drop("leag", axis=1, inplace=True)
# df.drop("gp", axis=1, inplace=True)
# df.drop("minutes", axis=1, inplace=True)
# df.drop("pts", axis=1, inplace=True)
# df.drop("oreb", axis=1, inplace=True)
# df.drop("dreb", axis=1, inplace=True)
# df.drop("reb", axis=1, inplace=True)
# df.drop("asts", axis=1, inplace=True)
# df.drop("stl", axis=1, inplace=True)
# df.drop("blk", axis=1, inplace=True)
# df.drop("turnover", axis=1, inplace=True)
df.drop("pf", axis=1, inplace=True)
df.drop("fga", axis=1, inplace=True)
df.drop("fgm", axis=1, inplace=True)
df.drop("fta", axis=1, inplace=True)
df.drop("ftm", axis=1, inplace=True)
df.drop("tpa", axis=1, inplace=True)
df.drop("tpm", axis=1, inplace=True)
df.dropna().head(20000).to_csv("data/nba.csv", sep=',', encoding='utf-8', index=False, header=False)


# oreb,continuous,
# dreb,continuous,
# reb,continuous,
# asts,continuous,
# stl,continuous,
# blk,continuous,
# turnover,continuous,
#
oreb
dreb
reb
asts
# stl
# blk
# turnover



