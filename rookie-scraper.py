import re
import os
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

def extract_info(li):
    a2 = li.find('a', {'href': re.compile('space')})
    return a2['href'].split('/')[-1]
    

url = 'https://www.bilibili.com/v/popular/rank/rookie'
resp = requests.get(url).content.decode('utf-8')
bs_page = BeautifulSoup(resp, 'lxml')
li = bs_page.find_all('li', {'class': 'rank-item'})
new_list = list(map(extract_info, li))
with open('rookie.txt', 'r') as f:
    old_list = f.read().split('\n')
    
new_list = list(filter(lambda x: x not in old_list, new_list))
rookie_list = old_list + new_list

with open('rookie.txt', 'w') as f:
    f.write('\n'.join(rookie_list))

try:
    os.mkdir('rookie_list')
except:
    pass

id_chunks = [[] for i in range(6)]
for i in range(len(rookie_list)):
    id_chunks[i%6].append(rookie_list[i])

for i in range(len(id_chunks)):
    index = i + 1
    chunk = id_chunks[i]
    with open(f'rookie_list/rookie{index}.txt', 'w') as f:
        f.write('\n'.join(chunk))
