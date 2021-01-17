import os
import re
import sys
import time
import opencc
import requests
import numpy as np
import numba as nb
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from bs4 import BeautifulSoup
from fake_headers import Headers


# when using "eval" function to convert json string to dict,
# it will treat these strings as variables
false = False
true = True
null = None

# text filter
T2S = opencc.OpenCC('t2s')
punc =  "[.!//_,$&%^*()<>+\"'?@#-:~{}]+[——\\\\=、：“”‘’《》【】￥……（）]+"
char = "[^\u4e00-\u9fa5^a-z^A-Z^0-9^，^。^！^？]"


def check_update(t_flag, up_list):
    '''
    intro: check if video producers upload new video after t_flag (timestamp)
    input:
        - t_flag (int/float)
        - up_list (list) contains id of interested video producers
    output:
        - pd.DataFrame (uid,bvid,cdate)
    '''
    update_list = []
    for i in up_list:
        # API of video publish list (mid = user id)
        url = f'https://api.bilibili.com/x/space/arc/search?mid={i}&ps=5&order=pubdate'
        # convert json string to python dict
        json = eval(requests.get(url).content.decode('utf-8'))
        try:
            # if there's no new video, this list would not have the key "vlist"
            bvid_list = [[i, v['bvid'], v['created']] 
                         for v in json['data']['list']['vlist'] if v['created'] > t_flag]
            for j in bvid_list:
                print(f'Update: {i}-{j[1]}')
        except:
            bvid_list = []
            print(f'Check Faild: {i}')
        update_list += bvid_list
        time.sleep(0.3)
    df_update = pd.DataFrame(update_list, columns=['uid', 'bvid', 'cdate'])
    return df_update


def get(url):
    '''
    intro: write "get" function. Automatically generate fake header information
    input:
        - target url (string)
    output:
        - response
    '''
    return requests.get(url, headers={'user-agent': Headers().generate()['User-Agent']})


def get_cid_via_bvid(bvid):
    '''
    intro: cid is required when scrapping danmaku
           every video has an unique bv id and an unique cid
    input: 
        - bvid (string)
    output:
        - cid (string)
    '''
    url = f'https://api.bilibili.com/x/player/pagelist?bvid={bvid}'
    resp = eval(requests.get(url).content.decode('utf-8'))
    return resp['data'][0]['cid']


def get_danmaku_page(cid):
    '''
    intro: access danmaku API with cid
    input:
        - cid (string)
    output:
        - danmaku page (html string)
    '''
    resp_search = get(f'http://comment.bilibili.com/{cid}.xml')
    page = resp_search.content.decode('utf-8')
    return page


def parse_danmaku_page(page):
    '''
    intro: extract data from danmaku page
    input:
        - danmaku page (html string)
    output:
        - pd.DataFrame
        [Sender,DmContent,AppearTime,SendTime,FontColor,DmType]
    '''
    table = list(map(
        lambda x: x.attrib['p'].split(',') + [x.text],
        ET.fromstring(page).findall('d')
    ))
    df = pd.DataFrame(table, columns=[
        'AppearTime', 'DmType', 'FontSize', 'FontColor', 
        'SendTime', 'DmAd', 'Sender', 'RowID', 'DmContent'
    ])
    return df[['Sender', 'DmContent', 'AppearTime', 
                'SendTime', 'FontColor', 'DmType']]


def get_danmaku(bvid):
    '''
    intro: get danmaku using bvid
    input:
        - bvid (string)
    output:
        - pd.DataFrame
        [Sender,DmContent,AppearTime,SendTime,FontColor,DmType]
    '''
    cid = get_cid_via_bvid(bvid)
    page = get_danmaku_page(cid)
    danmaku = parse_danmaku_page(page)
    return danmaku


def clean_text(text):
    '''
    intro: 
        - clean text data by filtering out some certain characters
        - this func will be used in the pandas "apply" func when 
          processing the column contains danmaku text data
    input:
        - origin text (string)
    output:
        - cleaned text (string)
    '''
    text = text.replace('|', '')
    text = re.sub(punc, '', text)
    text = re.sub(char, '', text)
    return T2S.convert(text)


@nb.jit(nopython=False)
def merge(sentence):
    '''
    intro: 
        - processing repeated characters
        - this func will truncate character sequence in which 
          a certain character repeats more than 4 times
        - this func will be used in the pandas "apply" func when 
          processing the column contains danmaku text data
    input:
        - character sequence (string)
    output:
        - cleaned text (string)
    '''
    max_ngram_length = 4
    final_merge_sent = sentence
    max_ngram_length = min(max_ngram_length, len(sentence))
    for i in range(max_ngram_length, 0, -1):
        start = 0
        end = len(final_merge_sent)-i+1
        ngrams = []
        while start < end:
            ngrams.append(final_merge_sent[start: start+i])
            start += 1
        result = []
        for cur_word in ngrams:
            result.append(cur_word)
            if len(result) > i:
                pre_word = result[len(result)-i-1]
                if pre_word == cur_word:
                    for k in range(i):
                        result.pop()

        cur_merge_sent = ''
        for word in result:
            if not cur_merge_sent:
                cur_merge_sent += word
            else:
                cur_merge_sent += word[-1]
        final_merge_sent = cur_merge_sent
    return final_merge_sent


def clean_danmaku(df):
    '''
    intro: clean danmaku text data
    input:
        - origin df (pd.DataFrame)
    output:
        - cleaned df (pd.DataFrame)
    '''
    df['AppearTime'] = df['AppearTime'].astype(float)
    df['SendTime'] = df['SendTime'].astype(int)
    df['DmContent'] = df['DmContent'].astype('str')
    df = df[df['DmContent'].apply(len)>1]
    df.loc[:,'DmContent'] = df['DmContent']\
                              .apply(clean_text)\
                              .apply(merge)\
                              .apply(lambda x: x.replace(' ', ''))
    df = df[df['DmContent'].apply(len)>1]
    df = df.sort_values('SendTime')
    df.reset_index(drop=True, inplace=True)
    return df

    
def check_and_scrape_dm(target_user, chunk):
    '''
    intro:
        - check if video producers upload new videos
        - scrape danmaku
    input:
        - target_user ("cyberstar", "rookie")
        - chunk index (1-6)
    create files:
        - video list (old+new)
        - danmaku file
    '''
    # create temporary folder to share files between jobs
    os.mkdir(f'{target_user}{chunk}')
    os.mkdir(f'{target_user}{chunk}/dm')
    os.mkdir(f'{target_user}{chunk}/update')
    
    # scrape danmaku within 2 weeks
    two_weeks_ago = int(datetime.now().timestamp()-14*24*60*60)

    # up_list will also be used in func "check_update"
    with open(f'./{target_user}_list/{target_user}{chunk}.txt', 'r') as f:
        up_list = f.read().split('\n')
    
    # init video list file
    file_name = f'{target_user}_update_list_{chunk}.csv'
    update_record_path = f'./{target_user}_list/{file_name}'
    if not os.path.exists(update_record_path):
        pd.DataFrame(columns=['uid', 'bvid', 'cdate'])\
          .to_csv(update_record_path, index=0)
    df = pd.read_csv(update_record_path)
    with open('date', 'r') as f:
        t_flag = int(f.read().strip())
        
    # check update
    df_update = check_update(t_flag, up_list)
    df = pd.concat([df, df_update])
    df.to_csv(f'./{target_user}{chunk}/update/{file_name}', index=0)
    
    # scrape danmaku
    for bvid in df['bvid'][df['cdate']>two_weeks_ago]:
        try:
            df_new = clean_danmaku(get_danmaku(bvid))
            df_new = df_new[df_new['SendTime']>t_flag]
            if len(df_new) > 0:
                print(f'Scrape DM: {bvid}')
            else:
                print(f'No updated DM: {bvid}')
        except KeyError:
            print(f'dm failed: {bvid}')
            continue
        df_old_path = f'./{target_user}_dm/{bvid}.csv'
        df_new_path = f'./{target_user}{chunk}/dm/{bvid}.csv'
        if os.path.exists(df_old_path):
            df_old = pd.read_csv(df_old_path)
        else:
            df_old = pd.DataFrame(columns=df_new.columns)
        if df_old['SendTime'].max() is np.nan:
            t_flag = 0
        else:
            t_flag = int(df_old['SendTime'].max())
        pd.concat([df_new, df_old])\
          .drop_duplicates()\
          .to_csv(df_new_path, index=0)


if __name__ == '__main__':
    check_and_scrape_dm(sys.argv[1], sys.argv[2])
