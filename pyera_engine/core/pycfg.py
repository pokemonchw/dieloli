# -*- coding: UTF-8 -*-
import os
import sys

if getattr(sys, 'frozen', False):
    # frozen
    dir_ = os.path.dirname(sys.executable)
    gamepath = os.path.dirname(dir_)
else:
    # unfrozen
    dir_ = os.path.dirname(os.path.realpath(__file__))
    gamepath = os.path.dirname(os.path.dirname(dir_))
print(dir_)
print(gamepath)
sys.path.append(gamepath)
print(gamepath)


platform = None
