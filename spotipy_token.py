import spotipy
import spotipy.util as util
import yaml

stream = open('config.yaml')
user_config = yaml.load(stream, Loader=yaml.FullLoader)
token = util.prompt_for_user_token(user_config['username'], 
        scope='playlist-read-private', 
        client_id=user_config['client_id'], 
        client_secret=user_config['client_secret'], 
        redirect_uri=user_config['redirect_uri'])
print(token)
spotipy.Spotify(auth=token)
