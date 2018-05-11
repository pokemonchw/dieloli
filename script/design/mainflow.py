# -*- coding: UTF-8 -*-
import os
import core.PyCmd as pycmd
import flow.CreatorPlayer as creatorplayer
import core.CacheContorl as cache
import flow.SaveHandleFrame as savehandleframe
import Panel.MainFlowPanel as mainflowpanel
import time
import design.MapHandle as maphandle

# 启动游戏界面
def open_func():
    mainflowpanel.loadGamePanel()
    time.sleep(1)
    path = maphandle.getPathfinding('0','1','3')
    print(path)
    #main_func()
    pass

# 主界面
def main_func():
    ans = mainflowpanel.gameMainPanel()
    if ans == 0:
        newgame_func()
    elif ans == 1:
        loadgame_func()
    elif ans == 2:
        quitgame_func()

# 主界面新建游戏调用
def newgame_func():
    pycmd.clr_cmd()
    cache.temporaryObject = cache.temporaryObjectBak.copy()
    creatorplayer.inputName_func()
    pass

# 主界面读取游戏调用
def loadgame_func():
    pycmd.clr_cmd()
    savehandleframe.loadSave_func('MainFlowPanel')
    pass

# 主界面退出游戏调用
def quitgame_func():
    os._exit(0)
    pass