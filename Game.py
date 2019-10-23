#!/usr/bin/python3
# -*- coding: UTF-8 -*-

from script.Core import GameData

GameData.init()

def gameStart():
    '''
    游戏启动函数
    '''
    from script.Design import StartFlow
    from script.Core import GameInit
    GameInit.run(StartFlow.startFrame)

gameStart()