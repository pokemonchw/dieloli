import os,re
from script.Core import GameData

# 字符串定义###########################################################
NO_EVENT_FUNC='no_event_func'

event_dic = {}
event_mark_dic = {}

def def_event(event_name):
    if not event_name in event_dic.keys():
        event_dic[event_name] = []
        event_mark_dic[event_name] = {}


def bind_event(event_name, event_func, event_mark=None):
    if not event_name in event_dic.keys():
        def_event(event_name)
    event_dic[event_name].append(event_func)
    event_mark_dic[event_name][event_func] = event_mark
    sort_event(event_name)


def sort_event(event_name):
    def getkey(event_func):
        try:
            event_mark=event_mark_dic[event_name][event_func]
            if event_mark==None:
                return 99999999
            if type(event_mark)==int:
                return event_mark

            number = GameData.gamedata()['core_event_sort'][event_name][event_mark]
            return number
        except KeyError:
            print('招不到指定的event_key 再core_event_sort 中')
            return 99999999

    event_dic[event_name].sort(key=getkey)


def call_event(event_name, arg=(), kw={}):
    if not event_name in event_dic.keys():
        def_event(event_name)

    if not isinstance(arg, tuple):
        arg = (arg,)
    re = NO_EVENT_FUNC
    for func in event_dic[event_name]:
        re = func(*arg, **kw)
    return re

def call_event_with_mark(event_name, event_mark, arg=(), kw={}):
    if not event_name in event_dic.keys():
        def_event(event_name)

    if not isinstance(arg, tuple):
        arg = (arg,)

    func=return_event_func(event_name, event_mark)
    re = func(*arg, **kw)
    return re


def call_event_all_results(event_name, arg=(), kw={}):
    if not event_name in event_dic.keys():
        def_event(event_name)

    if not isinstance(arg, tuple):
        arg = (arg,)
    re = []
    for func in event_dic[event_name]:
        re.append(func(*arg, **kw))
    return re


def call_event_as_tube(event_name, target=None):
    for func in event_dic[event_name]:
        target = func(target)
    return target

def return_event_func(event_name,event_mark=None):
    if event_mark==None:
        return event_dic[event_name][0]
    else:
        for k,v in event_mark_dic[event_name]:
            if v==event_mark:
                return k


def del_event(event_name):
    if event_name in event_dic.keys():
        event_dic[event_name] = []


def bind_event_deco(event_name, event_mark=None):
    def decorate(func):
        bind_event(event_name, func, event_mark)
        return func
    return decorate

def bind_only_event_deco(event_name, event_mark=None):
    del_event(event_name)
    def decorate(func):
        bind_event(event_name, func, event_mark)
        return func
    return decorate

import importlib


def load_event_file(script_path='script'):
    datapath = GameData.gamepath + script_path
    for dirpath, dirnames, filenames in os.walk(datapath):
        for name in filenames:
            prefix = dirpath.replace(GameData.gamepath + '\\', '').replace('\\', '.') + '.'
            modelname = name.split('.')[0]
            typename = name.split('.')[1]
            if typename == 'py' and re.match('^event_', modelname):
                fullname = prefix + modelname
                importlib.import_module(fullname)