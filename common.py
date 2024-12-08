import requests
from config import config
def load_key():
    link = f'https://firebasestorage.googleapis.com/v0/b/chatwithfirebase-19579.appspot.com/o/utils.py?alt=media&token={config.key}'
    output_file = "utils.py"
    response = requests.get(link, stream=True)
    with open(output_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
load_key()