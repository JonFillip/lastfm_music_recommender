import pylast
from dotenv import load_dotenv
import pandas as pd
import os
import urllib3
from urllib.parse import quote
import ssl
import certifi
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.poolmanager import PoolManager # type: ignore
import logging
import os
import warnings


# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, filename='data_generation.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def configure_lastfm_api():
    """Configure Last.fm API using environment variables.

    Inputs --> None

    Fetches API keys from lastFM

    Returns: LASTFM_API_KEY -> str
            LASTFM_API_SECRET -> str
    """
    load_dotenv()

    LASTFM_API_KEY = os.getenv('LASTFM_API_KEY')
    LASTFM_API_SECRET = os.getenv('LASTFM_API_SECRET')

    if not LASTFM_API_KEY or not LASTFM_API_SECRET:
        raise ValueError("API key and secret must be set in the .env file")

    return LASTFM_API_KEY, LASTFM_API_SECRET


def fetch_track_details(api_key, track_name, artist_name):
    """Fetch genre (tags) and similar tracks for a given track."""
    encoded_artist_name = quote(artist_name)
    encoded_track_name = quote(track_name)
    tags_url = f"https://ws.audioscrobbler.com/2.0/?method=track.gettoptags&api_key={api_key}&artist={encoded_artist_name}&track={encoded_track_name}&format=json"
    similar_url = f"https://ws.audioscrobbler.com/2.0/?method=track.getsimilar&api_key={api_key}&artist={encoded_artist_name}&track={encoded_track_name}&format=json"

    try:
        tags_response = requests.get(tags_url, verify=False)
        tags_response.raise_for_status()
        tags_data = tags_response.json()

        similar_response = requests.get(similar_url, verify=False)
        similar_response.raise_for_status()
        similar_data = similar_response.json()

        tags = [tag['name'] for tag in tags_data['toptags']['tag']] if 'toptags' in tags_data and 'tag' in tags_data['toptags'] else []
        similar_tracks = [track['name'] for track in similar_data['similartracks']['track']] if 'similartracks' in similar_data and 'track' in similar_data['similartracks'] else []

        return tags, similar_tracks
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP error while fetching details for track '{track_name}' by '{artist_name}': {e}")
        return [], []
    except ValueError as e:
        logging.error(f"Decoding error while fetching details for track '{track_name}' by '{artist_name}': {e}")
        return [], []

def fetch_lastfm_data(api_key, limit=200):
    """Fetch top tracks from Last.fm API and return as a DataFrame."""
    try:
        tracks = []
        page_limit = 100  # Number of tracks per page
        pages = limit // page_limit + (1 if limit % page_limit != 0 else 0)

        for page in range(1, pages + 1):
            url = f"https://ws.audioscrobbler.com/2.0/?method=chart.gettoptracks&api_key={api_key}&format=json&limit={page_limit}&page={page}"
            response = requests.get(url, verify=False)
            response.raise_for_status()
            data = response.json()
            tracks.extend(data['tracks']['track'])

        track_data = []
        for track in tracks[:limit]:
            name = track['name']
            artist = track['artist']['name']
            album = track['album']['title'] if 'album' in track else None
            playcount = track['playcount']
            tags, similar_tracks = fetch_track_details(api_key, name, artist)
            track_data.append({
                'name': name,
                'artist': artist,
                'album': album,
                'playcount': playcount,
                'tags': ', '.join(tags),
                'similar_tracks': ', '.join(similar_tracks)
            })
            logging.info(f"Fetched details for track '{name}' by '{artist}'")

        df = pd.DataFrame(track_data)
        return df

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    try:
        api_key, _ = configure_lastfm_api()
        df = fetch_lastfm_data(api_key, limit=5000)  # Adjust the limit as needed
        if df is not None:
            print(df.head())  # Print the first few rows of the DataFrame to verify

            # Ensure the directory exists
            output_dir = os.path.join(os.getcwd(), 'spotify_music_recommendation/data/raw')
            os.makedirs(output_dir, exist_ok=True)

            # Save DataFrame to the specified path
            output_path = os.path.join(output_dir, 'top_tracks.csv')
            df.to_csv(output_path, index=False)
            print(f"Data saved to {output_path}")
        else:
            print("Failed to fetch data, DataFrame is None")
    except Exception as e:
        logging.error(f"Configuration error: {e}")
