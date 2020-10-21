import pandas as pd
from youtubeAPIrequests import generate_youtube_statistics
import datetime

# 10000 quota limit from youtube, 
# 1000 quota cost for query, 
# 1 quota cost per specific video search
# buffer of 1
VIDEOS_LIMIT_PER_DAY = 10000 / (100 + 5) - 1
# VIDEOS_LIMIT_PER_DAY = 72

def main():
    start = datetime.datetime.now()
    movies_2010s = pd.read_csv("dbs/movies-from-2010s.csv")
    movies_2010s.dropna(inplace=True) # get rid of movies with null values
    movies_2010s = movies_2010s.loc[movies_2010s['revenue'] > 0] # get revenues that are greater than 0
    num_votes_required = movies_2010s['vote_count'].quantile(0.8)
    watched_movies_2010s = movies_2010s.copy().loc[movies_2010s['vote_count'] >= num_votes_required]
    watched_movies_2010s_tmdb_ids = watched_movies_2010s['tmdb_id'].values
    watched_movies_2010s.set_index('tmdb_id', inplace=True)
    release_dates_2010s = pd.read_csv("dbs/release_dates_2010s.csv").set_index('tmdb_id')
    cast_2010s = pd.read_csv("dbs/cast_2010s.csv")
    crew_2010s = pd.read_csv("dbs/crew_2010s.csv")
    
    trailers = []

    for i in range(VIDEOS_LIMIT_PER_DAY): # because of query limits, we must parse ths querying across multiple days
        # we need to generate a list of keywords for our youtube query filtering
        keywords = []
        tmdb_id = watched_movies_2010s_tmdb_ids[i]
        print(f"Iteration: {i} +++++ TMDB_ID: {tmdb_id}")
        release_date = release_dates_2010s.loc[tmdb_id, ['month_released', 'day_released',
            'year_released']]
        release_date = to_rfc3339(release_date)
        # first we want to care about highest-ordered actrs and characters
        actor_and_characters = cast_2010s.loc[cast_2010s['tmdb_id'] == tmdb_id].sort_values(
            by=['order'])[['actor_name', 'character']].head(5)
        # we also want to include the director
        directors = crew_2010s.loc[(crew_2010s['tmdb_id'] == tmdb_id) & (crew_2010s['job'] == 'Director'),
            'name'].values.tolist()
        keywords.extend(actor_and_characters['actor_name'].values.tolist())
        keywords.extend(actor_and_characters['character'].values.tolist())
        keywords.extend(directors)
        # now that we have our keywords and dates, we can query YouTube
        results = generate_youtube_statistics(tmdb_id, watched_movies_2010s.at[tmdb_id, 'title'],
            release_date, keywords)
        results = [vars(youtube_video) for youtube_video in results]
        trailers.extend(results)

    pd.DataFrame(trailers).to_csv("dbs/trailers_2010s.csv", index=False)

    print(f"TIME TO PROCESS ({VIDEOS_LIMIT_PER_DAY} Movies):", datetime.datetime.now()-start)

def to_rfc3339(release_date):
    '''
    convert to rfc3339 (1970-01-01T00:00:00Z) for youtube query
    Parameters
    ==========
    release_date:
        Pandas series object with columns year_released, month_released,
        and day_released
   
    Return
    ==========
    date in rfc3339 format
    '''

    return str(release_date['year_released']) + "-" \
                        + str(release_date['month_released']) + "-" \
                        + str(release_date['day_released']) \
                        + "T00:00:00Z"

if __name__ == "__main__":
    main()