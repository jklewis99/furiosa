import pandas as pd

def main():
    movies_2010s = pd.read_csv('movies-from-2010s.csv')
    movies_2010s.dropna(inplace=True) # get rid of movies without budget information

    # I want to use the 90th percentile for the number of votes required
    # to stay in the database
    num_votes_required = movies_2010s['vote_count'].quantile(0.9)
    # check how many votes are needed to be in the database
    print(num_votes_required)

    # apply this threshold to update the database
    watched_movies_2010s = movies_2010s.copy().loc[movies_2010s['vote_count'] >= num_votes_required]

    # check the size
    print(watched_movies_2010s.shape)
    # now we will generate video trailers movieIds with the youtube api 

if __name__ == "__main__":
    main()