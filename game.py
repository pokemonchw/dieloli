#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import getopt
import sys
from Script.Core import game_data


try:
    options = getopt.getopt(sys.argv[1:], "-d")[0]
    key = options[0][0]
    if key == "-d":
        game_data.init(1)
    else:
        game_data.init(0)
except:
    game_data.init(0)

from Script.Design import start_flow
from Script.Core import game_init
import Script.Talk
import Script.Behavior
import Script.Settle

game_init.run(start_flow.start_frame)
