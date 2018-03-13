# -*- coding: UTF-8 -*-
import os
import core.PyCmd as pycmd
import script.flow.CreatorPlayer as creatorplayer
import core.CacheContorl as cache
import script.flow.LoadSave as loadsave
import script.Panel.MainFlowPanel as mainflowpanel
import time

# 启动游戏界面
def open_func():
    mainflowpanel.loadGamePanel()
    time.sleep(1)
    main_func()
    pass

# 主界面
def main_func():
    mainflowpanel.gameMainPanel()
    pass

# 主界面新建游戏调用
def newgame_func():
    pycmd.clr_cmd()
    cache.temporaryObject = cache.temporaryObjectBak.copy()
    creatorplayer.inputName_func()
    pass

# 主界面读取游戏调用
def loadgame_func():
    pycmd.clr_cmd()
    loadsave.loadSave_func()
    pass

# 主界面退出游戏调用
def quitgame_func():
    os._exit(0)
    pass