#!/usr/bin/python3
# -*- coding: UTF-8 -*-

from Script.Core import game_data

game_data.init()

from Script.Design import start_flow
from Script.Core import game_init

game_init.run(start_flow.start_frame)
