import pandas as pd

# remove white spaces from the columns
df = pd.read_csv("starter/data/census.csv", sep=", ", engine="python")

# export to clean_data.csv
df.to_csv("starter/data/census_clean.csv", index=False)
