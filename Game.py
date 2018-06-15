# -*- coding: UTF-8 -*-
import sys
sys.path.append('script')
from core import GamePathConfig,game
from design import StartFlow

GamePathConfig.platform = 'win'

game.run(StartFlow.open_func)
