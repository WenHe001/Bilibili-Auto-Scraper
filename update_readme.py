import os
import requests
from datetime import datetime

false = False
true = True
null = None

url = 'https://api.github.com/repos/zijinghu/Bilibili-Auto-Scraper'
repo_info = eval(requests.get(url).content)
iso_time = repo_info['updated_at'].replace('Z', '+00:00')
delta = int(datetime.now().timestamp())

v_msg = f'# of videos:\t{int((len(os.listdir("cyberstar_dm"))-1)/2)}'
storage_msg = f'Repo Size:\t\t{str(round(repo_info["size"]/1024, 2))} MB'
update_msg = f'Timestamp:\t\t{str(delta)}'

with open('README.txt', 'w') as f:
    f.write('\n'.join([update_msg, storage_msg, v_msg]))