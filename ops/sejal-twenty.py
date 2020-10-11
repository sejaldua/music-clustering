import sys
import spotipy
import yaml
import spotipy.util as util
from pprint import pprint
import json

def load_config():
    global user_config
    stream = open('config.yaml')
    user_config = yaml.load(stream)
    # pprint(user_config)

def add_monthly_playlist_tracks(sources, target_playlist_id):
    for key in sources.keys():
        print(key)
        all_track_ids=[]
        all_track_names=[]
        track_list = sp.playlist_tracks(months[key])
        for track in track_list['items']:
            try:
                all_track_ids.append(track['track']['id'])
                all_track_names.append(track['track']['name'])
            except:
                continue
        # print(all_track_names)
        # print(all_track_ids)
        sp.user_playlist_add_tracks(user=user_config['username'], playlist_id=target_playlist_id, tracks=all_track_ids)
        print()

def playlist_uri_stripper(playlist_uri):
    return playlist_uri.split(':')[2]

if __name__ == '__main__':
    global sp
    global user_config
    months = {
        'jan20': '45hqvkXjiYcilhfj28Eydh',
        'feb20': '6FecIWptAeAPUbC8CxBjFu',
        'mar20': '44NeB9aqikV0Xm7vEjy7AX',
        'apr20': '6kogRNGaDDfIIV2c4AEwYB',
        'may20': '4mMRn8Kr8QhE2gZL0fcYoc',
        'jun20': '0NJ1vCjR453682Vl0UBW9i',
        'jul20': '16LJnglVuijORDhHZyc6hW',
        'aug20': '3a4MA8aVlqpcnrGtH1zPDX'
    }
    load_config()
    token = util.prompt_for_user_token(user_config['username'], scope='playlist-modify-private,playlist-modify-public', client_id=user_config['client_id'], client_secret=user_config['client_secret'], redirect_uri=user_config['redirect_uri'])
    if token:
        sp = spotipy.Spotify(auth=token)
        target_uri = input("Target playlist URI: ")
        target_id = playlist_uri_stripper(target_uri)
        tracks = add_monthly_playlist_tracks(months, target_uri)
    else:
        print ("Can't get token for", user_config['username'])