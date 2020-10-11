import sys
import spotipy
import yaml
import spotipy.util as util
from pprint import pprint
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_config():
    global user_config
    stream = open('config.yaml')
    user_config = yaml.load(stream, Loader=yaml.FullLoader)

def get_playlist_info(username, playlist_uri):
    playlist_id = uri.split(':')[2]
    results = sp.user_playlist(username, playlist_id)
    playlist_name = results['name']
    return playlist_name, results

def get_features_for_playlist(plists, username, uri, label):
    playlist_name, results = get_playlist_info(username, uri)
    plists[playlist_name] = {}
    plists[playlist_name]['name'] = []
    plists[playlist_name]['label'] = label
    plists[playlist_name]['track uri'] = []
    plists[playlist_name]['acousticness'] = []
    plists[playlist_name]['danceability'] = []
    plists[playlist_name]['energy'] = []
    plists[playlist_name]['instrumentalness'] = []
    plists[playlist_name]['liveness'] = []
    plists[playlist_name]['loudness'] = []
    plists[playlist_name]['speechiness'] = []
    plists[playlist_name]['tempo'] = []
    plists[playlist_name]['valence'] = []
    plists[playlist_name]['popularity'] = []

    for track in results['tracks']['items']:
        # print(json.dumps(track, indent=4))              # DEBUG STATEMENT
        
        # save metadata stuff
        name = track['track']['name']
        track_uri = track['track']['uri']
        plists[playlist_name]['name'].append(name)
        plists[playlist_name]['track uri'].append(track_uri)

        # extract features
        features = sp.audio_features(track_uri)
        plists[playlist_name]['acousticness'].append(features[0]['acousticness'])
        plists[playlist_name]['danceability'].append(features[0]['danceability'])
        plists[playlist_name]['energy'].append(features[0]['energy'])
        plists[playlist_name]['instrumentalness'].append(features[0]['instrumentalness'])
        plists[playlist_name]['liveness'].append(features[0]['liveness'])
        plists[playlist_name]['loudness'].append(features[0]['loudness'])
        plists[playlist_name]['speechiness'].append(features[0]['speechiness'])
        plists[playlist_name]['tempo'].append(features[0]['tempo'])
        plists[playlist_name]['valence'].append(features[0]['valence'])
    
    return plists


def print_audio_feature_stats(plists):
    """manually inspect all of the values to determine whether the median or mean is a better metric to plot"""
    for playlist in plists:
        print("––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––")
        print(playlist)
        for feature in plists[playlist]:
            if feature != 'name' and feature != 'track uri':
                print(feature.upper(), "| median:", np.median(plists[playlist][feature]), "| mean:", np.mean(plists[playlist][feature]))
        
# Helper function to plot each playlist on the radar chart.
def add_to_radar(ax, angles, pdict, color):

    values = [np.median(pdict['acousticness']), np.median(pdict['danceability']), np.median(pdict['energy']), 
            np.median(pdict['valence']), np.mean(pdict['instrumentalness']), np.median(pdict['tempo']), 
            np.median(pdict['speechiness'])]
    # tempo values typically range from 50-220, so I divided by 220 to get a number between 0 and 1
    values[-2] = values[-2]/220
    # speechiness values values are highly concentrated between 0 and 0.25-ish, so I multiplied by 4. Adjust this if needed
    values[-1] = values[-1]*4
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=1, label=pdict['label'])
    ax.fill(angles, values, color=color, alpha=0.25)
    return ax

def make_radar_graph(plists):

    # print_audio_feature_stats(plists)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    labels = ['acousticness', 'danceability', 'energy', 'valence', 'instrumentalness', 'tempo', 'speechiness']
    num_vars = len(labels)
    colors = ['red', 'green', 'blue']

    # Split the circle into even parts and save the angles so we know where to put each axis.
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Add each additional playlist to the chart.
    for i, pname in enumerate(plists.keys()):
        print(i, pname)
        ax = add_to_radar(ax, angles, plists[pname], colors[i])

    # polar coordinates math stuff
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(angles), labels)

    # Go through labels and adjust alignment based on where it is in the circle.
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
        
    # Set position of y-labels (0-100) to be in the middle of the first two axes.
    ax.set_ylim(0, 1)
    ax.set_rlabel_position(180 / num_vars)

    # Add some custom styling.
    ax.tick_params(colors='#222222')         # color of tick labels
    ax.tick_params(axis='y', labelsize=8)    # y-axis labels
    ax.grid(color='#AAAAAA')                 # color of circular gridlines
    ax.spines['polar'].set_color('#222222')  # color of outermost gridline (spine)
    ax.set_facecolor('#FAFAFA')              # background color inside the circle itself

    #Lastly, give the chart a title and a legend
    ax.set_title('Playlist Comparison', y=1.08)
    ax.legend(loc='best', bbox_to_anchor=(1.1, 1.1))

    fig.savefig('playlist_comp.png')

if __name__ == '__main__':
    global sp
    global user_config

    load_config()
    token = util.prompt_for_user_token(user_config['username'], scope='playlist-read-private', client_id=user_config['client_id'], client_secret=user_config['client_secret'], redirect_uri=user_config['redirect_uri'])
    if token:
        sp = spotipy.Spotify(auth=token)
        uris = []
        labels = []
        for i in range(2):
            uris.append(input("URI " + str(i+1) + ": "))
            labels.append(input("Label " + str(i+1) + ": "))
        plists = {}
        for i, uri in enumerate(uris):
            plists = get_features_for_playlist(plists, user_config['username'], uri, labels[i])
        make_radar_graph(plists)
    else:
        print ("Can't get token for", user_config['username'])