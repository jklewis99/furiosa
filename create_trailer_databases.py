'''
module to make the calls to YouTube Data API that are limited by quota limit
'''
import argparse
import datetime
import os
import pandas as pd
from similarity_score import get_similarity_score
from utils.misc import create_directory
from utils.helpers import clean_2010s_dataframe
from youtubeAPIrequests import search_youtube

# 10000 quota limit from youtube,
# 1000 quota cost for query,
# 1 quota cost per specific video search

def main():
    '''
    main method to make calls to the YouTube Data API as specified by the query-limit argument,
    offset by iteration-start, which define the indices of the watched_movies_2010s dataframe
    whose titles are the source of the query to YouTube search
    '''
    parser = argparse.ArgumentParser(description='Youtube querying based on tmdb film titles')
    parser.add_argument('input', type=str, help="Folder containing \
        movies-from-2010s, cast_2010s, crew_2010s, and release_dates_2010s csv files")
    parser.add_argument('output', type=str, help="Location to save \
        trailers_2010s_{iteration-start}.csv")
    parser.add_argument('-query-limit', '-ql', type=int, default=96, help="Maximum number of \
        search-and-video queries that can be made. A search-and-video query has a total unit \
        cost of 101 units.")
    parser.add_argument('-iteration-start', "-it", type=int, default=0, help="index on which \
        to start the loop of tmdb_ids")
    args = parser.parse_args()

    start = datetime.datetime.now()
    watched_movies_2010s = clean_2010s_dataframe(os.path.join(args.input, "movies_from_2010s.csv"))    
    movie_tmdb_ids = watched_movies_2010s['tmdb_id'].values
    movie_descriptions = watched_movies_2010s['overview'].values.tolist()
    watched_movies_2010s.set_index('tmdb_id', inplace=True)
    release_dates_2010s = pd.read_csv(f"{args.input}/release_dates_2010s.csv").set_index('tmdb_id')
    cast_2010s = pd.read_csv(f"{args.input}/cast_2010s.csv")
    crew_2010s = pd.read_csv(f"{args.input}/crew_2010s.csv")

    trailers = []
    errors = []

    # because of query limits, we must parse ths querying across multiple days
    iteration_end = min(len(movie_tmdb_ids), args.iteration_start + args.query_limit)
    for i in range(args.iteration_start, iteration_end):
        # we need to generate a list of keywords for our youtube query filtering
        keywords = []
        tmdb_id = movie_tmdb_ids[i]
        print(f"Iteration: {i} +++++ TMDB_ID: {tmdb_id}")

        release_date = release_dates_2010s.loc[
            tmdb_id, ['month_released', 'day_released', 'year_released']]
        release_date = to_rfc3339(release_date)

        # first we want to care about highest-ordered actors and characters
        actor_and_characters = cast_2010s.loc[cast_2010s['tmdb_id'] == tmdb_id].sort_values(
            by=['order'])[['actor_name', 'character']].head(5)
        # we also want to include the director
        directors = crew_2010s.loc[(crew_2010s['tmdb_id'] == tmdb_id) & (
            crew_2010s['job'] == 'Director'), 'name'].values.tolist()

        keywords.extend(actor_and_characters['actor_name'].values.tolist())
        keywords.extend(actor_and_characters['character'].values.tolist())
        keywords.extend(directors)

        # now that we have our keywords and dates, we can query YouTube
        title = watched_movies_2010s.at[tmdb_id, 'title']

        youtube_response = search_youtube(tmdb_id, title)
        results = get_similarity_score(title, youtube_response, movie_descriptions[i], keywords)
        results = [vars(youtube_video) for youtube_video in results]
        trailers.extend(results)

    create_directory(args.output)
    pd.DataFrame(trailers).to_csv(os.path.join(
        args.output, f"trailers_2010s_{args.iteration_start}.csv"), index=False)
    errors_file = open('errors/trailer-database-errors.txt', 'w+')
    errors_file.writelines(errors)
    print(f"TIME TO PROCESS ({iteration_end-args.iteration_start} Movies): \
        {datetime.datetime.now()-start}")

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
