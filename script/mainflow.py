# -*- coding: UTF-8 -*-
import core.game as game
import script.TextLoading as text
import os
import time

def open_func():
    game.p(text.loadMessageAdv(text.advGameLoadText))
    time.sleep(1)
    game.clr_screen()
    game.pl(text.loadMessageAdv(text.advGameIntroduce))
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
    game.pcmd(text.loadCmdAdv(text.cmdStartGameText), 1, newgame_func)
    pass

def loadgame_func():
    pass

def quitgame_func():
    os._exit(0)
    pass