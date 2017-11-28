# -*- coding: UTF-8 -*-
import core.game as game
import script.TextLoading as text
import script.GameConfig as config
import os
import time

def open_func():
    game.pobo(1/3,text.loadMessageAdv(text.advGameLoadText),)
    time.sleep(1)
    game.clr_screen()
    game.pti(config.game_name)
    game.p('\n')
    game.p('\n')
    game.p('\n')
    time.sleep(1)
    game.pobo(1/15,text.loadMessageAdv(text.advGameIntroduce))
    time.sleep(1)
    game.pline()
    time.sleep(1)
    main_func()
    pass

def main_func():
    game.pcmd(text.loadCmdAdv(text.cmdStartGameText),1,newgame_func)
    game.p('\n')
    game.pcmd(text.loadCmdAdv(text.cmdLoadGameText),2,loadgame_func)
    game.p('\n')
    game.pcmd(text.loadCmdAdv(text.cmdQuitGameText),3,quitgame_func)
    pass

def newgame_func():
    game.clr_cmd()
    pass

def loadgame_func():
    pass

def quitgame_func():
    os._exit(0)
    pass