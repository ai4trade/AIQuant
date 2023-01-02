# -*- coding: utf-8 -*-

import requests, time, datetime, hashlib

def _md5(data):
    return hashlib.md5(data.encode()).hexdigest()

def _sha1(data):
    return hashlib.sha1(data.encode()).hexdigest()

def get_sign(data):
    data = _sha1(data)
    data = _md5(data)
    return data

def get_cls_data(timestamp=None):
    if timestamp is None:
        timestamp = int(time.time())
    else:
        timestamp = int(timestamp)
    headers = {"Content-Type": "application/json;charset=utf-8", "Referer": "https://www.cls.cn/telegraph", "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"}
    data = 'app=CailianpressWeb&category=&lastTime={}&last_time={}&os=web&refresh_type=1&rn=20&sv=7.7.5'.format(timestamp, timestamp)
    url = 'https://www.cls.cn/nodeapi/telegraphList?' + data + "&sign=" + get_sign(data)
    resp = requests.get(url, headers=headers)
    return resp.json()

print(get_cls_data(timestamp=datetime.datetime.fromisoformat('2023-01-01 00:00:00').timestamp()))