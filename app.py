import spotipy
import yaml
import spotipy.util as util
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
sns.set()
import altair as alt
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from pathlib import Path
from kneed import KneeLocator
import streamlit as st
from math import sqrt


def main():
    df = pd.DataFrame(columns=['name', 'artist', 'track_URI', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'])
    playlist_uri = st.text_input("Please enter a playlist URI", 'spotify:playlist:2U4FxOBn1Ga2INJbD3AGBu')
    playlist_name, df = get_features_for_playlist(df, user_config['username'], playlist_uri)
    st.write("### " + playlist_name, df)
    # page = st.sidebar.selectbox("Choose a page", ["Homepage", "Mechanical", "Environmental"])
    # x_axis = list(df['name'])
    # y_axis = st.selectbox("Choose a variable for the y-axis", list(df.columns)[3:], index=2)
    # visualize_data(df, x_axis, y_axis)
    optimal_k_graph, (ax1, ax2), clustered_df = kmeans(df)
    st.write(optimal_k_graph)
    visualize_clusters(clustered_df)


def visualize_clusters(df):
    graph = alt.Chart(df.reset_index()).mark_circle(size=60).encode(
        x=alt.X('Component 2'),
        y=alt.Y('Component 1'),
        color=alt.Color('Cluster', scale=alt.Scale(scheme='category20b')),
        tooltip=['name', 'artist']
    ).interactive()

    st.altair_chart(graph, use_container_width=True)

# Get Spotipy credentials from config
def load_config():
    stream = open('config.yaml')
    user_config = yaml.load(stream, Loader=yaml.FullLoader)
    return user_config

@st.cache(allow_output_mutation=True)
def get_token(user_config):
    token = util.prompt_for_user_token(user_config['username'], 
        scope='playlist-read-private', 
        client_id=user_config['client_id'], 
        client_secret=user_config['client_secret'], 
        redirect_uri=user_config['redirect_uri'])
    return spotipy.Spotify(auth=token)

# A function to extract track names and URIs from a playlist
def get_playlist_info(username, playlist_uri):
    # initialize vars
    offset = 0
    tracks, uris, names, artists = [], [], [], []

    # get playlist id and name from URI
    playlist_id = playlist_uri.split(':')[2]
    playlist_name = sp.user_playlist(username, playlist_id)['name']

    # get all tracks in given playlist (max limit is 100 at a time --> use offset)
    while True:
        results = sp.user_playlist_tracks(username, playlist_id, offset=offset)
        tracks += results['items']
        if results['next'] is not None:
            offset += 100
        else:
            break
        
    # get track metadata
    for track in tracks:
        names.append(track['track']['name'])
        artists.append(track['track']['artists'][0]['name'])
        uris.append(track['track']['uri'])
    
    return playlist_name, names, artists, uris

@st.cache(allow_output_mutation=True)
def get_features_for_playlist(df, username, uri):
    # get all track metadata from given playlist
    playlist_name, names, artists, uris = get_playlist_info(username, uri)
    
    # iterate through each track to get audio features and save data into dataframe
    for name, artist, track_uri in zip(names, artists, uris):
        # print(json.dumps(track_uri, indent=4))              
        # ^ DEBUG STATEMENT ^
        
        # access audio features for given track URI via spotipy 
        audio_features = sp.audio_features(track_uri)

        # get relevant audio features
        feature_subset = [audio_features[0][col] for col in df.columns if col not in ["name", "artist", "track_URI"]]

        # compose a row of the dataframe by flattening the list of audio features
        row = [name, artist, track_uri, *feature_subset]
        df.loc[len(df.index)] = row
    return playlist_name, df

def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 1

def visualize_data(df, x_axis, y_axis):
    graph = alt.Chart(df.reset_index()).mark_bar().encode(
        x=alt.X('name', sort='y'),
        y=alt.Y(str(y_axis)+":Q"),
    ).interactive()

    st.altair_chart(graph, use_container_width=True)

def num_components_graph(ax, num_columns, evr):
    ax.plot(range(1, num_columns+1), evr.cumsum(), 'bo-')
    ax.set_title('Explained Variance by Components')
    ax.set(xlabel='Number of Components', ylabel='Cumulative Explained Variance')
    ax.hlines(0.8, xmin=1, xmax=num_columns, linestyles='dashed')
    return ax

def num_clusters_graph(ax, max_clusters, wcss):
    ax.plot([i for i in range(1, max_clusters)], wcss, 'bo-')
    ax.set_title('Optimal Number of Clusters')
    ax.set(xlabel='Number of Clusters [k]', ylabel='Within Cluster Sum of Squares (WCSS)')
    ax.vlines(KneeLocator([i for i in range(1, max_clusters)], wcss, curve='convex', direction='decreasing').knee, ymin=0, ymax=max(wcss), linestyles='dashed')
    return ax

@st.cache(allow_output_mutation=True)
def kmeans(df):
    df_X = df.drop(columns=df.columns[:3])
    print("Standard scaler and PCA")
    scaler = StandardScaler()
    X_std = scaler.fit_transform(df_X) 
    pca = PCA()
    pca.fit(X_std)
    evr = pca.explained_variance_ratio_
    for i, exp_var in enumerate(evr.cumsum()):
        if exp_var >= 0.8:
            n_comps = i + 1
            break
    print("Finding optimal number of components", n_comps)
    pca = PCA(n_components=n_comps)
    pca.fit(X_std)
    scores_pca = pca.transform(X_std)
    wcss = []
    max_clusters = 11
    for i in range(1, max_clusters):
        kmeans_pca = KMeans(i, init='k-means++', random_state=42)
        kmeans_pca.fit(scores_pca)
        wcss.append(kmeans_pca.inertia_)
    n_clusters = KneeLocator([i for i in range(1, max_clusters)], wcss, curve='convex', direction='decreasing').knee
    print("Finding optimal number of clusters", n_clusters)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1 = num_components_graph(ax1, len(df_X.columns), evr)
    ax2 = num_clusters_graph(ax2, max_clusters, wcss)
    print("Performing KMeans")
    kmeans_pca = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans_pca.fit(scores_pca)
    df_seg_pca_kmeans = pd.concat([df_X.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
    df_seg_pca_kmeans.columns.values[(-1 * n_comps):] = ["Component " + str(i+1) for i in range(n_comps)]
    df_seg_pca_kmeans['Cluster'] = kmeans_pca.labels_
    df['Cluster'] = df_seg_pca_kmeans['Cluster']
    df['Component 1'] = df_seg_pca_kmeans['Component 1']
    df['Component 2'] = df_seg_pca_kmeans['Component 2']
    fig.tight_layout()
    return fig, (ax1, ax2), df


if __name__ == "__main__":
    user_config = load_config()
    sp = get_token(user_config)
    # Initialize Spotify API token
    
    main()