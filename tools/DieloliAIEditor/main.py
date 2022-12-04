#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import sys
import json
import os
from PySide6.QtWidgets import QApplication,QFileDialog
from PySide6.QtGui import QKeySequence, QShortcut
from ui.window import Window
from ui.menu_bar import MenuBar
from ui.state_machine_tree import StateMachineTree
from ui.data_list import DataList
import load_csv
import cache_control
import json_handle
import game_type

load_csv.load_config()
app = QApplication(sys.argv)
main_window: Window = Window()
menu_bar: MenuBar = MenuBar()
state_machine_tree: StateMachineTree = StateMachineTree()
data_list: DataList = DataList()


def load_ai_data():
    """载入ai数据"""
    now_file = QFileDialog.getOpenFileName(menu_bar,"选择文件",".","*.json")
    file_path = now_file[0]
    if file_path:
        cache_control.now_file_path = file_path
        now_data = json_handle.load_json(file_path)
        for k in now_data:
            now_target: game_type.Target = game_type.Target()
            now_target.__dict__ = now_data[k]
            cache_control.now_target_data[k] = now_target


def create_ai_data():
    """新建ai数据"""
    dialog = QFileDialog(menu_bar)
    dialog.setFileMode(QFileDialog.AnyFile)
    dialog.setNameFilter("Json (*.json)")
    if dialog.exec():
        file_names = dialog.selectedFiles()
        file_path: str = file_names[0]
        if not file_path.endswith(".json"):
            file_path += ".json"
        cache_control.now_file_path = file_path


def save_ai_data():
    """保存ai数据"""
    if cache_control.now_file_path:
        with open(cache_control.now_file_path,"w",encoding="utf-8") as ai_data_file:
            now_data = {}
            for k in cache_control.now_target_data:
                now_data[k] = cache_control.now_target_data[k].__dict__
            json.dump(now_data, ai_data_file, ensure_ascii=0)


def exit_editor():
    """关闭编辑器"""
    os._exit(0)


def clicked_state_machine(item: game_type.TreeItem,column:int):
    """
    点击状态机列表是更新目标列表
    Keyword arguments:
    item -- 点击的对象
    column -- 点击的位置
    """
    if "cid" not in item.__dict__:
        return
    cache_control.now_state_machine_id = item.cid
    data_list.update()
    menu_bar.status_menu.setTitle("当前状态机:"+cache_control.state_machine_data[item.cid])


menu_bar.select_ai_file_action.triggered.connect(load_ai_data)
menu_bar.new_ai_file_action.triggered.connect(create_ai_data)
menu_bar.save_ai_action.triggered.connect(save_ai_data)
menu_bar.exit_action.triggered.connect(exit_editor)
state_machine_tree.itemActivated.connect(clicked_state_machine)
state_machine_tree.setFixedWidth(600)
main_window.setMenuBar(menu_bar)
main_window.add_widget(state_machine_tree)
main_window.add_widget(data_list)
main_window.completed_layout()
QShortcut(QKeySequence(main_window.tr("Ctrl+O")),main_window,load_ai_data)
QShortcut(QKeySequence(main_window.tr("Ctrl+N")),main_window,create_ai_data)
QShortcut(QKeySequence(main_window.tr("Ctrl+S")),main_window,save_ai_data)
QShortcut(QKeySequence(main_window.tr("Ctrl+Q")),main_window,exit_editor)
main_window.show()
app.exec()
