import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import spotipy

from spotipy.oauth2 import SpotifyOAuth

# set open_browser=False to prevent Spotipy from attempting to open the default browser
spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(show_dialog=True))

print(spotify.me())
