# -*- coding: UTF-8 -*-
import core.EraPrint as eprint
import script.TextLoading as text
import core.GameConfig as config
import os
import time
import core.PyCmd as pycmd
import core.pyio as pyio
import script.flow.CreatorPlayer as creatorplayer

def open_func():
    eprint.pobo(1/3,text.loadMessageAdv('1'))
    time.sleep(1)
    pyio.clear_screen()
    eprint.pti(config.game_name)
    eprint.p('\n')
    eprint.p('\n')
    eprint.p('\n')
    time.sleep(1)
    eprint.lcp(1/8,text.loadMessageAdv('2'))
    time.sleep(1)
    eprint.pline()
    time.sleep(1)
    main_func()
    pass

def main_func():
    pycmd.focusCmd()
    pycmd.pcmd(text.loadCmdAdv('1'),1,newgame_func)
    eprint.p('\n')
    pycmd.pcmd(text.loadCmdAdv('2'),2,loadgame_func)
    eprint.p('\n')
    pycmd.pcmd(text.loadCmdAdv('3'),3,quitgame_func)
    pass

def newgame_func():
    pycmd.clr_cmd()
    eprint.pnextscreen()
    creatorplayer.inputName_func()
    pass

def loadgame_func():
    pycmd.clr_cmd()
    pass

def quitgame_func():
    os._exit(0)
    pass