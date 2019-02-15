#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from script.Core import GameData

GameData.init()
def gameStart():
    from script.Design import StartFlow
    from script.Core import GameInit
    GameInit.run(StartFlow.open_func)

gameStart()
