# -*- coding: UTF-8 -*-
import sys
sys.path.append('script')
from Core import GamePathConfig,GameInit
from Design import StartFlow

GamePathConfig.platform = 'win'

GameInit.run(StartFlow.open_func)
