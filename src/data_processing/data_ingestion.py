import os
import asyncio
import aiohttp
from dotenv import load_dotenv
import pandas as pd
from urllib.parse import quote
from src.utils.logging_utils import get_logger
from google.cloud import bigquery
from cachetools import TTLCache
from concurrent.futures import ThreadPoolExecutor

logger = get_logger('data_ingestion')

# Cache for storing track details
cache = TTLCache(maxsize=10000, ttl=3600)

def configure_lastfm_api():
    """Configure Last.fm API using environment variables."""
    load_dotenv()
    LASTFM_API_KEY = os.getenv('LASTFM_API_KEY')
    LASTFM_API_SECRET = os.getenv('LASTFM_API_SECRET')
    if not LASTFM_API_KEY or not LASTFM_API_SECRET:
        raise ValueError("API key and secret must be set in the .env file")
    return LASTFM_API_KEY, LASTFM_API_SECRET

async def fetch_track_details(session, api_key, track_name, artist_name):
    """Fetch genre (tags) and similar tracks for a given track."""
    cache_key = f"{artist_name}:{track_name}"
    if cache_key in cache:
        return cache[cache_key]

    encoded_artist_name = quote(artist_name)
    encoded_track_name = quote(track_name)
    tags_url = f"https://ws.audioscrobbler.com/2.0/?method=track.gettoptags&api_key={api_key}&artist={encoded_artist_name}&track={encoded_track_name}&format=json"
    similar_url = f"https://ws.audioscrobbler.com/2.0/?method=track.getsimilar&api_key={api_key}&artist={encoded_artist_name}&track={encoded_track_name}&format=json"

    try:
        async with session.get(tags_url) as tags_response, session.get(similar_url) as similar_response:
            tags_data = await tags_response.json()
            similar_data = await similar_response.json()

        tags = [tag['name'] for tag in tags_data['toptags']['tag']] if 'toptags' in tags_data and 'tag' in tags_data['toptags'] else []
        similar_tracks = [track['name'] for track in similar_data['similartracks']['track']] if 'similartracks' in similar_data and 'track' in similar_data['similartracks'] else []

        result = (tags, similar_tracks)
        cache[cache_key] = result
        return result
    except Exception as e:
        logger.error(f"Error fetching details for track '{track_name}' by '{artist_name}': {e}")
        return [], []

async def fetch_lastfm_data(api_key, limit=200):
    """Fetch top tracks from Last.fm API and return as a DataFrame."""
    async with aiohttp.ClientSession() as session:
        try:
            tracks = []
            page_limit = 100  # Number of tracks per page
            pages = limit // page_limit + (1 if limit % page_limit != 0 else 0)

            for page in range(1, pages + 1):
                url = f"https://ws.audioscrobbler.com/2.0/?method=chart.gettoptracks&api_key={api_key}&format=json&limit={page_limit}&page={page}"
                async with session.get(url) as response:
                    data = await response.json()
                    tracks.extend(data['tracks']['track'])

            track_data = []
            tasks = []
            for track in tracks[:limit]:
                name = track['name']
                artist = track['artist']['name']
                album = track['album']['title'] if 'album' in track else None
                playcount = track['playcount']
                tasks.append(fetch_track_details(session, api_key, name, artist))

            results = await asyncio.gather(*tasks)

            for track, (tags, similar_tracks) in zip(tracks[:limit], results):
                track_data.append({
                    'name': track['name'],
                    'artist': track['artist']['name'],
                    'album': track['album']['title'] if 'album' in track else None,
                    'playcount': track['playcount'],
                    'tags': ', '.join(tags),
                    'similar_tracks': ', '.join(similar_tracks)
                })
                logger.info(f"Fetched details for track '{track['name']}' by '{track['artist']['name']}'")

            df = pd.DataFrame(track_data)
            return df

        except Exception as e:
            logger.error(f"An error occurred while fetching Last.fm data: {e}")
            return pd.DataFrame()

def write_to_bigquery(df, project_id, dataset_id, table_id):
    """Write DataFrame to BigQuery table."""
    client = bigquery.Client(project=project_id)
    table_ref = client.dataset(dataset_id).table(table_id)
    job_config = bigquery.LoadJobConfig()
    job_config.autodetect = True
    job_config.source_format = bigquery.SourceFormat.CSV

    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()  # Wait for the job to complete

    logger.info(f"Loaded {job.output_rows} rows into {dataset_id}:{table_id}")

async def main(project_id, dataset_id, table_id):
    try:
        api_key, _ = configure_lastfm_api()
        df = await fetch_lastfm_data(api_key, limit=5000)  # Adjust the limit as needed
        if not df.empty:
            logger.info(f"Successfully fetched {len(df)} tracks from Last.fm")
            write_to_bigquery(df, project_id, dataset_id, table_id)
        else:
            logger.error("Failed to fetch data, DataFrame is empty")
    except Exception as e:
        logger.error(f"Error in main function: {e}")

if __name__ == '__main__':
    project_id = 'your-project-id'  # Replace with your GCP project ID
    dataset_id = 'lastfm_dataset'
    table_id = 'top_tracks'
    asyncio.run(main(project_id, dataset_id, table_id))
