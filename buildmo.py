#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import os

po_dir_path = os.path.join("data","po")
po_dir_list = os.listdir(po_dir_path)
for language_id in po_dir_list:
    po_dir = os.path.join(po_dir_path,language_id,"LC_MESSAGES")
    po_path = os.path.join(po_dir, "dieloli.po")
    mo_path = os.path.join(po_dir, "dieloli.mo")
    os.system("msgfmt " + po_path + " -o " + mo_path)
    print(language_id,"Building End")
print("Mo Building End")
