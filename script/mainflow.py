# -*- coding: UTF-8 -*-
import core.EraPrint as eprint
import script.TextLoading as textload
import core.TextHandle as text
import core.GameConfig as config
import os
import time
import core.PyCmd as pycmd
import core.pyio as pyio
import script.flow.CreatorPlayer as creatorplayer
import core.CacheContorl as cache

def open_func():
    eprint.pobo(1/3,textload.loadMessageAdv('1'))
    time.sleep(1)
    pyio.clear_screen()
    eprint.pline()
    eprint.pl(text.align(config.game_name,'center'))
    eprint.pl(text.align(config.author,'right'))
    eprint.pl(text.align(config.verson,'right'))
    eprint.pline()
    eprint.p('\n')
    eprint.lcp(1/8,textload.loadMessageAdv('2'))
    time.sleep(1)
    eprint.p('\n' * 4)
    eprint.pline()
    time.sleep(1)
    main_func()
    pass

def main_func():
    pycmd.focusCmd()
    eprint.p('\n')
    pycmd.pcmd(textload.loadCmdAdv('1'),1,newgame_func)
    eprint.p('\n')
    pycmd.pcmd(textload.loadCmdAdv('2'),2,loadgame_func)
    eprint.p('\n')
    pycmd.pcmd(textload.loadCmdAdv('3'),3,quitgame_func)
    pass

def newgame_func():
    pycmd.clr_cmd()
    eprint.pnextscreen()
    cache.temporaryObject = cache.temporaryObjectBak
    creatorplayer.inputName_func()
    pass

def loadgame_func():
    pycmd.clr_cmd()
    pass

def quitgame_func():
    os._exit(0)
    pass