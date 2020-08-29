#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import getopt
import sys
from Script.Config import game_config


game_config.init()


from Script.Design import start_flow, handle_target, handle_premise
from Script.Core import game_init
import Script.Talk
import Script.Settle
import Script.UI.Panel

game_init.run(start_flow.start_frame)
