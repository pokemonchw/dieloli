#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import getopt
import sys
from Script.Core import game_data


try:
    options, _ = getopt.getopt(sys.argv[1], "d:", ["debug"])
    key, value = options[0]
    if key in ("-d", "--debug"):
        game_data.init(1)
    else:
        game_data.init(0)
except:
    game_data.init(0)

from Script.Design import start_flow
from Script.Core import game_init
import Script.Talk.default.default

game_init.run(start_flow.start_frame)
