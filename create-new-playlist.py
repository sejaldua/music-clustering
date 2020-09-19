import sys
import spotipy
import yaml
import spotipy.util as util
from pprint import pprint
import json
import argparse


def load_config():
    global user_config
    stream = open('config.yaml')
    user_config = yaml.load(stream, Loader=yaml.FullLoader)

def create_new_playlist(user, name, description):
    info = sp.user_playlist_create(user=user, name=name, public=True, collaborative=False, description=description)
    return info['id']

if __name__ == '__main__':
    global sp
    global user_config

    parser = argparse.ArgumentParser()
    #-n NAME -d DESCRIPTION -c COLLABORATIVE -p PUBLIC
    parser.add_argument("-n", "--name", help="Playlist name")
    parser.add_argument("-d", "--description", help="Playlist description")
    args = parser.parse_args()

    print( "NAME: \t\t{} \nDESCRIPTION \t{} ".format(
            args.name,
            args.description,
            ))

    load_config()
    token = util.prompt_for_user_token(user_config['username'], scope='playlist-modify-private,playlist-modify-public', client_id=user_config['client_id'], client_secret=user_config['client_secret'], redirect_uri=user_config['redirect_uri'])
    if token:
        sp = spotipy.Spotify(auth=token)
        uri = create_new_playlist(user_config['username'], args.name, args.description)
        print(uri)
    else:
        print ("Can't get token for", user_config['username'])