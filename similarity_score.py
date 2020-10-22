from youtube_video import YouTubeVideo
import textdistance

def strip_characters(title):
    replaced = ""
    for char in title.lower():
        if char == " " or char == "\\" or char == "/":
            replaced += " "
        else:
            val = ord(char)
            if val > 60 and val < 123 or val > 47 and val < 58:
                replaced += char
    return replaced

def description_keywords(description_list, keywords):
    '''
    add scores based on keywords in the description

    Parameters
    ==========
    description_list:
        list of words in the description of the video
    keywords:
        list of keywords, such as director and cast members

    Return
    ==========
    integer of count of tose keywords in the description
    '''
    count_contained = 0
    if len(description_list) == 0 or keywords is None:
        return 0
    for word in keywords:
        if word.lower() in description_list:
            count_contained += 1
    
    return count_contained / len(keywords)

def is_clip(trailer_title_tokens, movie_title_tokens, 
    flags=["clip", "clips", "scene", "scenes", "blooper", "bloopers"]):
    '''
    checks if any terms in flags are in the trailer title list.
    if these flags are contained in the movie_title_tokens, then
    the method defaults to returning 1

    Parameters
    ==========
    trailer_title_tokens:
        list of lowercase tokens in trailer title
    movie_title_tokens:
        list of lowercase tokens in movie title
    
    Return
    ==========
    -1 if flags are in trailer title and not in movie title, 1 otherwise
    '''

    # TODO: if this {flag}-in-movie-title is a severe issue, it would be smarter to check the 
    #       specific flag in the trailer and ensure it is surrounded by the same words as
    #       the words that surround it in the title
    if any(item in movie_title_tokens for item in flags):
        return 1
    contains_flag = any(item in trailer_title_tokens for item in flags)
    if contains_flag:
        return -1
    return 1

def get_similarity_score(title, videos, movie_description, keywords=None):
    '''
    main method that take the list of video_data from a YouTube query
    and sort the queries based on a set of similarities as defined
    by accord index and description keywords

    Parameters
    ==========
    title:
        the title of the movie
    videos:
        list of youtube video objects
    keywords:
        list of top-5 cast, top-5 characters, and director(s)
    movie_description:
        string of description from tmdb

    Return
    ==========
    list of YouTubeVideo objects with similarity score
    '''

    youtube_videos = []
    for video in videos:
        # first, all items will be converted to lowercase and tokenized
        lowercase_title = title.lower()
        movie_desc = strip_characters(movie_description.lower()).split()
        trailer_desc = strip_characters(video.description.lower()).split()
        movie_title = (strip_characters(lowercase_title) + " official trailer").split()
        trailer_title = strip_characters(video.title.lower()).split()

        similarity_score = is_clip(trailer_title, lowercase_title.split()) * 0.5 * \
            (textdistance.sorensen_dice(movie_desc, trailer_desc)
            + textdistance.jaccard(movie_title, trailer_title))
        video.set_similarity_score(similarity_score)
        youtube_videos.append(video)

    return youtube_videos

def test():
    s1 = "prince of persia: the sands of time"
    desc = "A rogue prince reluctantly joins forces with a mysterious princess and together, \
        they race against dark forces to safeguard an ancient dagger capable of releasing the \
            Sands of Time â€“ gift from the gods that can reverse time and allow its possessor \
                to rule the world."

    trailers = ["Prince of Persia:The Sands of Time - Trailer Movie (HQ)",
            "Prince Of Persia Sands Of Time Trailer HD",
            "Prince of Persia: The Sands of Time Trailer C",
            "Prince of Persia: The Sands of Time - Official Trailer 2",
            "PRINCE OF PERSIA: THE SANDS OF TIME MOVIE TRAILER"]
    descriptions = [
        "Prince of Persia: The Sands of Time - Trailer Movie Walt Disney May 2010 \
            By: www.ign.com real Movie, Videogame Film Adaptation. \
            See more movies and videogames in www.ign.com \
            upload by: karloromeo",
        "JUST TO REMEMBER OLD TIMES ^^ \
            Prince of persia sands of time game trailer enjoy \
            i will upload prince of persia 4/2008 soon!",
        "Prince of Persia - The Sands of Time is vanaf 1 september verkrijgbaar op Disney Blu-ray en DVD!",
        "Click Here To Purchase: http://bit.ly/d230RB \
        (Blu-ray/DVD Combo + Digital Copy)  in stores September 14, 2010. \
        Become a fan of PRINCE OF PERSIA on Facebook at: http://facebook.com/PrinceOfPersiaMovie \
        Set in the mystical lands of Persia, PRINCE OF PERSIA: THE SANDS OF TIME is an epic \
        action-adventure about a rogue prince (JAKE GYLLENHAAL) and a mysterious princess \
        (GEMMA ARTERTON) who race against dark forces to safeguard an ancient dagger \
            capable of releasing the Sands of Timeâ€”a gift from the gods that can \
                 reverse time and allow its possessor to rule the world.",
                 "Click Here To Purchase: http://bit.ly/d230RB \
        (Blu-ray/DVD Combo + Digital Copy)  in stores September 14, 2010. \
        Become a fan of PRINCE OF PERSIA on Facebook at: http://facebook.com/PrinceOfPersiaMovie \
        Set in the mystical lands of Persia, PRINCE OF PERSIA: THE SANDS OF TIME is an epic \
        action-adventure about a rogue prince (JAKE GYLLENHAAL) and a mysterious princess \
        (GEMMA ARTERTON) who race against dark forces to safeguard an ancient dagger \
            capable of releasing the Sands of Timeâ€”a gift from the gods that can \
                 reverse time and allow its possessor to rule the world."
    ]
    videos = []
    for i in range(5):
        obj = YouTubeVideo('', trailers[i], '', '', descriptions[i],'', '', '', '', '', '')
        videos.append(obj)

    import pandas as pd
    tmdb_id = 78105
    cast_2010s = pd.read_csv("dbs/cast_2010s.csv")
    crew_2010s = pd.read_csv("dbs/crew_2010s.csv")

    actor_and_characters = cast_2010s.loc[cast_2010s['tmdb_id'] == tmdb_id].sort_values(
            by=['order'])[['actor_name', 'character']].head(5)
    # we also want to include the director
    directors = crew_2010s.loc[(crew_2010s['tmdb_id'] == tmdb_id) & (crew_2010s['job'] == 'Director'),
        'name'].values.tolist()

    keywords = []
    keywords.extend(actor_and_characters['actor_name'].values.tolist())
    keywords.extend(actor_and_characters['character'].values.tolist())
    keywords.extend(directors)

    ugh = get_similarity_score(s1, videos, desc)
    for i in ugh:
        print(i.similarity_score)

if __name__ == "__main__":
    test()