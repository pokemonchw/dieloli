#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from script.Core import GamePathConfig,GameInit
from script.Design import StartFlow

GamePathConfig.platform = 'win'

GameInit.run(StartFlow.open_func)
