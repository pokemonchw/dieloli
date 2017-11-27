# -*- coding: UTF-8 -*-
import core.pycfg

core.pycfg.platform = 'web'

import core.game
import script.mainflow

core.game.run(script.mainflow.open_func)
