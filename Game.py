# -*- coding: UTF-8 -*-
import sys
sys.path.append('script')
import core.pycfg

core.pycfg.platform = 'win'
import core.game

script = __import__('design.mainflow')

core.game.run(script.mainflow.open_func)
