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

v_msg1 = f'# of videos (C):\t{int((len(os.listdir("cyberstar_dm"))-1)/2)}'
v_msg2 = f'# of videos (R):\t{int((len(os.listdir("rookie_dm"))-1)/2)}'
storage_msg = f'Repo Size:\t{str(round(repo_info["size"]/1024, 2))} MB'
update_msg = f'Timestamp:\t{str(delta)}'

with open('README.txt', 'w') as f:
    f.write('\n'.join([update_msg, storage_msg, v_msg1, v_msg2]))
