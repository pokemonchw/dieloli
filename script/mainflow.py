# -*- coding: UTF-8 -*-
import core.EraPrint as eprint
import script.TextLoading as textload
import core.TextHandle as text
import core.GameConfig as config
import os
import time
import core.PyCmd as pycmd
import script.flow.CreatorPlayer as creatorplayer
import core.CacheContorl as cache
import script.Ans as ans
import  script.flow.LoadSave as loadsave
import core.flow as flow

# 启动游戏界面
def open_func():
    eprint.pnextscreen()
    flow.initCache()
    eprint.pobo(1 / 3, textload.loadMessageAdv('1'))
    eprint.p('\n')
    time.sleep(1)
    main_func()
    pass

# 主界面
def main_func():
    eprint.pline()
    eprint.pl(text.align(config.game_name, 'center'))
    eprint.pl(text.align(config.author, 'right'))
    eprint.pl(text.align(config.verson, 'right'))
    eprint.pl(text.align(config.verson_time,'right'))
    eprint.p('\n')
    eprint.pline()
    eprint.lcp(1/3,textload.loadMessageAdv('2'))
    time.sleep(1)
    eprint.p('\n')
    eprint.pline()
    time.sleep(1)
    pycmd.focusCmd()
    menuInt = ans.optionint(ans.logomenu)
    if menuInt == 0:
        newgame_func()
    elif menuInt == 1:
        loadgame_func()
    elif menuInt == 2:
        quitgame_func()
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