# -*- coding: UTF-8 -*-
from core.pycfg import gamepath
import json
try:
    import cPickle as pickle
except ImportError:
    import pickle

import os

_gamedata = {}

def gamedata():
    return _gamedata

def is_utf8bom(pathfile):
    if b'\xef\xbb\xbf' == open(pathfile, mode='rb').read(3):
        return True
    return False

def _loadjson(filepath):
    if is_utf8bom(filepath):
        ec='utf-8-sig'
    else:
        ec='utf-8'
    with open(filepath, 'r', encoding=ec) as f:
        try:
            jsondata = json.loads(f.read())
        except json.decoder.JSONDecodeError:
            print(filepath + '  无法读取，文件可能不符合json格式')
            jsondata = []
    return jsondata

def _loaddir(datapath):
    for dirpath, dirnames, filenames in os.walk(datapath):
        for name in filenames:
            thefilepath = os.path.join(dirpath,name)
            prefix = dirpath.replace(datapath, '').replace('\\', '.') + '.'
            if prefix == '.':
                prefix = ''
            if name.split('.')[1] == 'json':
                _gamedata[prefix + name.split('.')[0]] = _loadjson(thefilepath)

def init():
    global gamepath
    datapath = os.path.join(gamepath,'data')
    _loaddir(datapath)

def _get_savefilename_path(filename):
    global gamepath
    savepath = os.path.join(gamepath,'save')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    filepath = os.path.join(savepath,filename + '.save')
    return filepath

def save(filename, data=None):
    if data == None:
        data = _gamedata
    filepath = _get_savefilename_path(filename)
    # with open(filepath, 'w', encoding='utf-8') as f:
        # json.dump(data, f, ensure_ascii=False)
    with open(filepath, 'wb') as f:
        pickle.dump(data,f)

def load(filename, selfdata=False):
    filepath = _get_savefilename_path(filename)
    data = {}
    try:
        # with open(filepath, 'r', encoding='utf-8') as f:
            # data = json.load(f)
        with open(filepath, 'rb') as f:
            data=pickle.load(f)
    except FileNotFoundError:
        print(filepath + '  没有该存档文件')
    if selfdata == False:
        global _gamedata
        _gamedata.update(data)
    return data