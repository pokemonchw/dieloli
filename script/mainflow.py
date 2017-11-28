# -*- coding: UTF-8 -*-
import core.EraPrint as eprint
import script.TextLoading as text
import core.GameConfig as config
import os
import time
import core.PyCmd as pycmd
import core.pyio as pyio

def open_func():
    eprint.pobo(1/3,text.loadMessageAdv(text.advGameLoadText),)
    time.sleep(1)
    pyio.clear_screen()
    eprint.pti(config.game_name)
    eprint.p('\n')
    eprint.p('\n')
    eprint.p('\n')
    time.sleep(1)
    eprint.pobo(1/15,text.loadMessageAdv(text.advGameIntroduce))
    time.sleep(1)
    eprint.pline()
    time.sleep(1)
    main_func()
    pass

def main_func():
    pycmd.pcmd(text.loadCmdAdv(text.cmdStartGameText),1,newgame_func)
    eprint.p('\n')
    pycmd.pcmd(text.loadCmdAdv(text.cmdLoadGameText),2,loadgame_func)
    eprint.p('\n')
    pycmd.pcmd(text.loadCmdAdv(text.cmdQuitGameText),3,quitgame_func)
    pass

def newgame_func():
    pycmd.clr_cmd()
    pass

def loadgame_func():
    pass

def quitgame_func():
    os._exit(0)
    pass