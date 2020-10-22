import pandas as pd
from tmdbAPIrequests import appended_movie_info
import datetime

def main():
    start = datetime.datetime.now()
    movies_2010s = pd.read_csv("dbs/movies-from-2010s.csv")
    movies_2010s.dropna(inplace=True) # get rid of movies with null values
    movies_2010s = movies_2010s.loc[movies_2010s['revenue'] > 0] # get revenues that are greater than 0
    num_votes_required = movies_2010s['vote_count'].quantile(0.8)
    watched_movies_2010s = movies_2010s.copy().loc[movies_2010s['vote_count'] >= num_votes_required]
    watched_movies_2010s_tmdb_ids = watched_movies_2010s['tmdb_id'].values

    # TODO: better source for reviews, TMDB is limited in these
    
    release_dates = []
    crew = []
    cast = []
    
    for tmdb_id in watched_movies_2010s_tmdb_ids:
        response = appended_movie_info(tmdb_id)
        # reponse contains keys "credits", "reviews", and "release_date"
        release_dates.append(update_date(response['release_date'], tmdb_id))
        cast.extend(update_credits(response['credits']['cast'], tmdb_id))
        crew.extend(update_credits(response['credits']['crew'], tmdb_id))
    cast_2010s = pd.DataFrame(cast)[['tmdb_id', 'name', 'character', 'id', 'cast_id', 'credit_id', 'order', 'gender']]
    cast_2010s.rename(columns={'id': 'tmdb_person_id', 'name': 'actor_name'}, inplace=True)
    cast_2010s.to_csv("cast_2010s.csv", index=False)
    crew_2010s = pd.DataFrame(crew)[['tmdb_id', 'name', 'job', 'department', 'id', 'credit_id', 'gender']]
    crew_2010s.rename(columns={'id': 'tmdb_person_id'}, inplace=True)
    crew_2010s.to_csv("crew_2010s.csv", index=False)
    pd.DataFrame(release_dates).to_csv("release_dates_2010s.csv", index=False)
    print("TIME TO PROCESS (677 Movies):", datetime.datetime.now()-start) #0:01:40.366053

def update_credits(credits, tmdb_id):
    '''
    method to add the tmdb_id key to a dictionary instance of a cast/crew member

    Parameters
    ==========
    credits: 
        a list (either cast or crew) of dictionaries
    tmdb_id: 
        the tmdb id of the movie

    Returns
    ==========
    updated list with tmdb_id as a key/value pair for each dictionary
    '''
    credits_members = []
    for credit in credits:
        credit.update(tmdb_id=tmdb_id)
        credits_members.append(credit)
    return credits_members

def update_date(release_date, tmdb_id):
    '''
    method to parse out a date object in format: YYYY-MM-DDT00:00:00.000Z

    Parameters
    ==========
    release_date:
        a string in format YYYY-MM-DDT00:00:00.000Z
    tmdb_id:
        the tmdb id of the movie

    Returns
    ==========
    a dictionary of tmdb_id, weekday, month, day, and year (of movies release)
    '''
    if release_date is None:
        return {
            "tmdb_id": tmdb_id,
            "weekday_released": "",
            "month_released": 0, 
            "day_released": 0, 
            "year_released": 0,
        }
    split_date = [int(x) for x in release_date[:10].split("-")]
    date = datetime.datetime(split_date[0], split_date[1], split_date[2])
    return {
        "tmdb_id": tmdb_id,
        "weekday_released": date.strftime("%A"),
        "month_released": date.month, 
        "day_released": date.day, 
        "year_released": date.year,
        }
    
if __name__ == "__main__":
    main()