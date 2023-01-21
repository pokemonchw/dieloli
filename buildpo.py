#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import os

po_dir_path = os.path.join("data","po")
po_dir_list = os.listdir(po_dir_path)
po_dir = os.path.join(po_dir_path,"zh_CN","LC_MESSAGES")
po_path = os.path.join(po_dir, "dieloli.po")
if os.path.exists(po_path):
    os.remove(po_path)
os.system('find ./ -name "*.py" >POTFILES && xgettext -n --files-from=POTFILES -o ' + po_path)
os.remove("POTFILES")
print("Po Building End")
