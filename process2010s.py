import pandas as pd 

movies_2010s = pd.read_csv('dbs/movies-from-2010s.csv')
movies_2010s.dropna(inplace=True) # get rid of movies without budget information

# I want to use the 90th percentile for the number of votes required
# to stay in the database
num_votes_required = movies_2010s['vote_count'].quantile(0.9)
print(num_votes_required)

# apply this threshold to update the database
watched_movies_2010s = movies_2010s.copy().loc[movies_2010s['vote_count'] >= num_votes_required]

watched_movies_2010s.sort_values(by="budget", ascending=False, inplace=True)
print(watched_movies_2010s[['tmdb_id', 'title']].head(12)) #, 'vote_count', 'vote_average', 'budget', 'year', 'revenue']].head(20))