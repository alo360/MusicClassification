# -*- coding: utf-8 -*-

# Sample Python code for youtube.search.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/guides/code_samples#python
# %run setAPIkey.py

import os

# import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from tubestats.setAPIkey import YT_API_KEY

# scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]

def search_key_word(keyword='blues'):
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    # os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    # client_secrets_file = "tubestats\client_secret_29919533702-8ba1mp2u4epjbsfgcou0bj1fo4qbo5ft.apps.googleusercontent.com.json"

    # Get credentials and create an API client
    # flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
    #     client_secrets_file, scopes)
    # credentials = flow.run_console()
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=YT_API_KEY)
        #credentials=credentials)

    request = youtube.search().list(
        # part="snippet",
        part="id",
        order="viewCount",
        maxResults=1,
        q=keyword+" song",
        videoDuration="any"
    )
    response = request.execute()
    return response
    # print(response)

# if __name__ == "__main__":
#     main()