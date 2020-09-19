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

def add_playlist_tracks(src_playlist_id, target_playlist_id):
    all_track_ids=[]
    all_track_names=[]
    track_list = sp.playlist_tracks(src_playlist_id)
    for track in track_list['items']:
        try:
            all_track_ids.append(track['track']['id'])
            all_track_names.append(track['track']['name'])
        except:
            continue
    # print(all_track_names)
    # print(all_track_ids)
    sp.user_playlist_add_tracks(user=user_config['username'], playlist_id=target_playlist_id, tracks=all_track_ids)

def playlist_uri_stripper(playlist_uri):
    return playlist_uri.split(':')[2]

if __name__ == '__main__':
    global sp
    global user_config

    load_config()
    token = util.prompt_for_user_token(user_config['username'], scope='playlist-modify-private,playlist-modify-public', client_id=user_config['client_id'], client_secret=user_config['client_secret'], redirect_uri=user_config['redirect_uri'])
    if token:
        sp = spotipy.Spotify(auth=token)

        target_uri_raw = input("Which playlist do you want to add to / migrate songs to?\n")
        flag = True
        while flag:
            try:
                target_id = playlist_uri_stripper(target_uri_raw)
                flag = False
            except:
                print("Oops that didn't work. Please go to the playlist > ... > Share > Copy Spotify URI")
                target_uri_raw = input("Try again!\n")


        source_uri = input("Please enter as many source Spotify playlist URIs as you want. Enter 'Done' when you are finished.\n")
        while source_uri != "Done":
            try: 
                src_id = playlist_uri_stripper(source_uri)
                add_playlist_tracks(src_id, target_id)
                source_uri = input()
            except:
                print("Error... exiting")
                break
    else:
        print ("Can't get token for", user_config['username'])