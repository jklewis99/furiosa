
# Python code for getting the statistics of a movie trailer
# TODO: location moved; requires testing
import os
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors

from api_info.api_variables import api_service_name, api_version

def get_video_statistics(videoId, youtube):
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    request = youtube.videos().list(
        part="statistics, snippet",
        id=videoId
    )
    response = request.execute()

    print(response)

def main():
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version)
    get_video_statistics('EXeTwQWrcwY', youtube)

if __name__ == "__main__":
    main()