import spotipy
import os
import spotipy.util as util
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import altair as alt
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from kneed import KneeLocator
import streamlit as st
from math import sqrt
from matplotlib import cm
import SessionState
from spotipy.oauth2 import SpotifyClientCredentials

session_state = SessionState.get(checkboxed=False, num=2)
columns = ['name', 'artist', 'track_URI', 'playlist', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence']

def main():
    st.markdown("## Welcome to Playlist Blendr :wave:")
    st.markdown("### This web app uses machine learning techniques to cluster music by similar audio features so that you can cultivate a cohesive vibe to satisfy your listening needs!")
    num_playlists = st.sidebar.number_input('How many playlists would you like to cluster?', 1, 5, 2)
    if session_state.num != num_playlists:
        session_state.num = num_playlists
        session_state.checkboxed = False
    playlists = playlist_user_input(num_playlists)
    if st.sidebar.button("Run Algorithm") or session_state.checkboxed:
        session_state.checkboxed = True
        print(playlists)
        df = concatenate_playlists(playlists)
        if df is None:
            st.warning("One of your playlist URIs was not entered properly")
            st.stop()
        else:
            st.write(df)

            clustered_df, n_clusters = kmeans(df)
            print(clustered_df['playlist'].value_counts())        

            cluster_labels = clustered_df['Cluster']
            orig = clustered_df.drop(columns=['Cluster', "Component 1", "Component 2"])
            orig.insert(4, "cluster", cluster_labels)
            norm_df = make_normalized_df(orig, 5)
            fig, maxes = make_radar_chart(norm_df, n_clusters)
            st.write(fig)

            range_ = get_color_range(n_clusters)
            visualize_clusters(clustered_df, n_clusters, range_)

            explore_df = orig.copy()
            keys = sorted(list(explore_df["cluster"].unique()))
            cluster = st.selectbox("Choose a cluster to preview", keys, index=0)
            preview_df = preview_cluster_playlist(explore_df, cluster)
            st.write(preview_df[preview_df.columns[:5]])
            x_axis = list(preview_df['name'])
            y_axis = st.selectbox("Choose a variable for the y-axis", list(preview_df.columns)[5:], index=maxes[cluster])
            visualize_data(preview_df, x_axis, y_axis, n_clusters, range_)
    else:
        pass

def playlist_user_input(num_playlists):
    playlists = []
    defaults = ["spotify:playlist:37i9dQZF1DX9UhtB5CtZ7e", "spotify:playlist:37i9dQZF1DWSP55jZj2ES3",
    "spotify:playlist:37i9dQZF1DX4OzrY981I1W",
    "spotify:playlist:37i9dQZF1DX8dTWjpijlub",
    "spotify:playlist:37i9dQZF1DWUE76cNNotSg"
    ]
    st.sidebar.write("To locate a playlist URI, go to the playlist on Spotify, click the '...' button at the top, then go to Share > Copy Spotify URI. Some examples are pre-filled :)")
    for i in range(num_playlists):
        playlists.append(st.sidebar.text_input("Playlist URI " + str(i+1), defaults[i]))
    return playlists

def concatenate_playlists(playlists):
    global columns
    print("concatenate playlists")
    df = pd.DataFrame(columns=columns)
    if all(playlists):
        for playlist_uri in playlists:
            df = pd.concat([df, get_features_for_playlist(os.environ.get('USERNAME'), playlist_uri)], ignore_index=True, axis=0)
        return df
    else:
        return None

# Get Spotipy credentials from config
def load_config():
    stream = open('config.yaml')
    user_config = yaml.load(stream, Loader=yaml.FullLoader)
    return user_config

@st.cache(allow_output_mutation=True)
def get_token():
    print("generating token")
    # token = util.prompt_for_user_token(
    #     username=os.environ.get('USERNAME'),
    #     scope='playlist-read-private', 
    #     client_id=os.environ.get('CLIENT_ID'), 
    #     client_secret=os.environ.get('CLIENT_SECRET'), 
    #     redirect_uri=os.environ.get('REDIRECT_URI'))
    # sp = spotipy.Spotify(auth=token)
    client_credentials_manager = SpotifyClientCredentials(client_id=os.environ.get('CLIENT_ID'), client_secret=os.environ.get('CLIENT_SECRET'))
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return sp

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
def get_features_for_playlist(username, uri):
    # initialize_df
    global columns
    temp_df = pd.DataFrame(columns=columns)

    # get all track metadata from given playlist
    playlist_name, names, artists, uris = get_playlist_info(username, uri)
    
    # iterate through each track to get audio features and save data into dataframe
    for name, artist, track_uri in zip(names, artists, uris):
        
        # access audio features for given track URI via spotipy 
        audio_features = sp.audio_features(track_uri)

        # get relevant audio features
        feature_subset = [audio_features[0][col] for col in temp_df.columns if col not in ["name", "artist", "track_URI", "playlist"]]

        # compose a row of the dataframe by flattening the list of audio features
        row = [name, artist, track_uri, playlist_name, *feature_subset]
        temp_df.loc[len(temp_df.index)] = row
    return temp_df

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

def visualize_data(df, x_axis, y_axis, n_clusters, range_):
    graph = alt.Chart(df.reset_index()).mark_bar().encode(
        x=alt.X('name', sort='y'),
        y=alt.Y(str(y_axis)+":Q"),
        color=alt.Color('cluster', scale=alt.Scale(domain=[i for i in range(n_clusters)], range=range_)),
        tooltip=['name', 'artist']
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
    df_X = df.drop(columns=df.columns[:4])
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
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1 = num_components_graph(ax1, len(df_X.columns), evr)
    # ax2 = num_clusters_graph(ax2, max_clusters, wcss)
    print("Performing KMeans")
    kmeans_pca = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans_pca.fit(scores_pca)
    df_seg_pca_kmeans = pd.concat([df_X.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
    df_seg_pca_kmeans.columns.values[(-1 * n_comps):] = ["Component " + str(i+1) for i in range(n_comps)]
    df_seg_pca_kmeans['Cluster'] = kmeans_pca.labels_
    df['Cluster'] = df_seg_pca_kmeans['Cluster']
    df['Component 1'] = df_seg_pca_kmeans['Component 1']
    df['Component 2'] = df_seg_pca_kmeans['Component 2']
    # fig.tight_layout()
    return df, n_clusters

@st.cache(allow_output_mutation=True)
def get_color_range(n_clusters):
    cmap = cm.get_cmap('tab20b')    
    range_ = []
    for i in range(n_clusters):
        color = 'rgb('
        mapped = cmap(i/n_clusters)
        for j in range(3):
            color += str(int(mapped[j] * 255))
            if j != 2:
                color += ", "
            else:
                color += ")"
        range_.append(color)
    return range_

def visualize_clusters(df, n_clusters, range_):
    graph = alt.Chart(df.reset_index()).mark_point(filled=True, size=60).encode(
        x=alt.X('Component 2'),
        y=alt.Y('Component 1'),
        shape=alt.Shape('playlist', scale=alt.Scale(range=["circle", "diamond", "square", "triangle-down", "triangle-up"])),
        color=alt.Color('Cluster', scale=alt.Scale(domain=[i for i in range(n_clusters)], range=range_)),
        tooltip=['name', 'artist']
    ).interactive()
    st.altair_chart(graph, use_container_width=True)

@st.cache(allow_output_mutation=True)
def make_normalized_df(df, col_sep):
    print(len(df))
    non_features = df[df.columns[:col_sep]]
    features = df[df.columns[col_sep:]]
    norm = MinMaxScaler().fit_transform(features)
    scaled = pd.DataFrame(norm, index=df.index, columns = df.columns[col_sep:])
    return pd.concat([non_features, scaled], axis=1)

@st.cache(allow_output_mutation=True)
def make_radar_chart(norm_df, n_clusters):
    fig = go.Figure()
    cmap = cm.get_cmap('tab20b')
    angles = list(norm_df.columns[5:])
    angles.append(angles[0])

    layoutdict = dict(
                radialaxis=dict(
                visible=True,
                range=[0, 1]
                ))
    maxes = dict()

    for i in range(n_clusters):
        subset = norm_df[norm_df['cluster'] == i]
        data = [np.mean(subset[col]) for col in angles[:-1]]
        maxes[i] = data.index(max(data))
        data.append(data[0])
        fig.add_trace(go.Scatterpolar(
            r=data,
            theta=angles,
            # fill='toself',
            # fillcolor = 'rgba' + str(cmap(i/n_clusters)),
            mode='lines',
            line_color='rgba' + str(cmap(i/n_clusters)),
            name="Cluster " + str(i)))
        
    fig.update_layout(
            polar=layoutdict,
            showlegend=True
    )
    fig.update_traces()
    return fig, maxes

@st.cache(allow_output_mutation=True)
def preview_cluster_playlist(df, cluster):
    df = df[df['cluster'] == cluster]

    # if st.button("Export to playlist"):
    #     result = sp.user_playlist_create(user_config['username'], 'cluster'+str(cluster), public=True, collaborative=False, description='')
    #     playlist_id = result['id']
    #     songs = list(df.loc[df['cluster'] == cluster]['track_URI'])
    #     if len(songs) > 100:
    #         sp.playlist_add_items(playlist_id, songs[:100])
    #         sp.playlist_add_items(playlist_id, songs[100:])
    #     else:
    #         sp.playlist_add_items(playlist_id, songs)
    # else:
    #     pass
    return df

if __name__ == "__main__":
    # user_config = load_config()
    
    # Initialize Spotify API token
    sp = get_token()
    main()