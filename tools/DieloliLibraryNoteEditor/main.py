#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import json
import uuid
import csv
from typing import Dict, Any

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QListWidget, QListWidgetItem, QTextEdit,
    QSplitter, QPushButton, QMessageBox, QLabel, QLineEdit,
    QSpinBox, QComboBox, QTreeWidget, QTreeWidgetItem, QDialog, QDialogButtonBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt

def load_csv_data(filepath, key_index, value_index=None):
    data = {}
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                if len(row) > key_index:
                    key = row[key_index]
                    val = row[value_index] if value_index is not None and len(row) > value_index else key
                    data[key] = val
    return data

class PremiseDialog(QDialog):
    def __init__(self, parent=None, premise_data=None):
        super().__init__(parent)
        self.setWindowTitle("选择前提条件")
        self.premise_data = premise_data or {}
        layout = QVBoxLayout(self)
        self.combo = QComboBox()
        for k, v in self.premise_data.items():
            self.combo.addItem(f"{k} ({v})", k)
        self.val_spin = QSpinBox()
        self.val_spin.setRange(0, 999999)
        self.val_spin.setValue(1)
        layout.addWidget(QLabel("前提条件:"))
        layout.addWidget(self.combo)
        layout.addWidget(QLabel("数值:"))
        layout.addWidget(self.val_spin)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self):
        return self.combo.currentData(), self.val_spin.value()

class EffectDialog(QDialog):
    def __init__(self, parent=None, effect_data=None):
        super().__init__(parent)
        self.setWindowTitle("选择行为效果")
        self.effect_data = effect_data or {}
        layout = QVBoxLayout(self)
        self.combo = QComboBox()
        for k, v in self.effect_data.items():
            self.combo.addItem(f"{k} ({v})", k)
        self.val_spin = QSpinBox()
        self.val_spin.setRange(0, 999999)
        self.val_spin.setValue(1)
        layout.addWidget(QLabel("行为效果:"))
        layout.addWidget(self.combo)
        layout.addWidget(QLabel("数值:"))
        layout.addWidget(self.val_spin)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self):
        return self.combo.currentData(), self.val_spin.value()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图书馆夹带物配置编辑器")
        self.resize(800, 600)
        self.config_data = {}
        
        self.premise_dict = load_csv_data("../premise.csv", 0, 2)
        self.effect_dict = load_csv_data("../Settle.csv", 3, 2)
        
        self.current_uid = None
        self.init_ui()
        self.load_data()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel: list of notes
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        left_layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        btn_add = QPushButton("新增")
        btn_add.clicked.connect(self.add_note)
        btn_del = QPushButton("删除")
        btn_del.clicked.connect(self.del_note)
        btn_save = QPushButton("保存")
        btn_save.clicked.connect(self.save_data)
        
        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_del)
        btn_layout.addWidget(btn_save)
        left_layout.addLayout(btn_layout)

        splitter.addWidget(left_widget)

        # Right panel: details
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # UID
        uid_layout = QHBoxLayout()
        uid_layout.addWidget(QLabel("UID:"))
        self.uid_edit = QLineEdit()
        self.uid_edit.setReadOnly(True)
        uid_layout.addWidget(self.uid_edit)
        right_layout.addLayout(uid_layout)

        # Type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("类型 (Type):"))
        self.type_combo = QComboBox()
        self.type_combo.addItem("留言 (0)", 0)
        self.type_combo.addItem("漂流瓶 (1)", 1)
        type_layout.addWidget(self.type_combo)
        right_layout.addLayout(type_layout)

        # Related Knowledge
        rk_layout = QHBoxLayout()
        rk_layout.addWidget(QLabel("关联书籍/知识库 (Related Knowledge):"))
        self.rk_spin = QSpinBox()
        self.rk_spin.setRange(0, 9999)
        rk_layout.addWidget(self.rk_spin)
        right_layout.addLayout(rk_layout)

        # Content Template
        right_layout.addWidget(QLabel("文本内容模板 (Content Template):"))
        self.content_edit = QTextEdit()
        right_layout.addWidget(self.content_edit)

        # Premise Conditions
        premise_layout = QHBoxLayout()
        premise_layout.addWidget(QLabel("前提条件 (Premise Conditions):"))
        btn_add_premise = QPushButton("+")
        btn_add_premise.clicked.connect(self.add_premise)
        btn_del_premise = QPushButton("-")
        btn_del_premise.clicked.connect(self.del_premise)
        premise_layout.addWidget(btn_add_premise)
        premise_layout.addWidget(btn_del_premise)
        right_layout.addLayout(premise_layout)
        
        self.premise_tree = QTreeWidget()
        self.premise_tree.setHeaderLabels(["前提", "数值"])
        right_layout.addWidget(self.premise_tree)

        # Action Effects
        effect_layout = QHBoxLayout()
        effect_layout.addWidget(QLabel("行为效果 (Action Effects):"))
        btn_add_effect = QPushButton("+")
        btn_add_effect.clicked.connect(self.add_effect)
        btn_del_effect = QPushButton("-")
        btn_del_effect.clicked.connect(self.del_effect)
        effect_layout.addWidget(btn_add_effect)
        effect_layout.addWidget(btn_del_effect)
        right_layout.addLayout(effect_layout)
        
        self.effect_tree = QTreeWidget()
        self.effect_tree.setHeaderLabels(["效果", "数值"])
        right_layout.addWidget(self.effect_tree)

        btn_apply = QPushButton("应用修改")
        btn_apply.clicked.connect(self.apply_changes)
        right_layout.addWidget(btn_apply)

        splitter.addWidget(right_widget)
        splitter.setSizes([200, 600])

    def load_data(self):
        if os.path.exists("default.json"):
            with open("default.json", "r", encoding="utf-8") as f:
                try:
                    self.config_data = json.load(f)
                except json.JSONDecodeError:
                    self.config_data = {}
        else:
            self.config_data = {}
        self.update_list()

    def update_list(self):
        self.list_widget.clear()
        for uid, data in self.config_data.items():
            item = QListWidgetItem(f"[{uid}] {data.get('type', 0)}")
            item.setData(Qt.ItemDataRole.UserRole, uid)
            self.list_widget.addItem(item)

    def on_item_clicked(self, item):
        uid = item.data(Qt.ItemDataRole.UserRole)
        self.current_uid = uid
        data = self.config_data[uid]
        
        self.uid_edit.setText(uid)
        
        idx = self.type_combo.findData(data.get("type", 0))
        if idx >= 0:
            self.type_combo.setCurrentIndex(idx)
        
        self.rk_spin.setValue(data.get("related_knowledge", 0))
        self.content_edit.setPlainText(data.get("content_template", ""))
        
        self.premise_tree.clear()
        for p, v in data.get("premise_condition", {}).items():
            QTreeWidgetItem(self.premise_tree, [str(p), str(v)])
            
        self.effect_tree.clear()
        for e, v in data.get("action_effect", {}).items():
            QTreeWidgetItem(self.effect_tree, [str(e), str(v)])

    def add_note(self):
        uid = str(uuid.uuid4())
        self.config_data[uid] = {
            "uid": uid,
            "type": 0,
            "related_knowledge": 0,
            "content_template": "",
            "premise_condition": {},
            "action_effect": {}
        }
        self.update_list()

    def del_note(self):
        if self.current_uid in self.config_data:
            del self.config_data[self.current_uid]
            self.current_uid = None
            self.update_list()

    def add_premise(self):
        dlg = PremiseDialog(self, self.premise_dict)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            k, v = dlg.get_data()
            QTreeWidgetItem(self.premise_tree, [str(k), str(v)])

    def del_premise(self):
        items = self.premise_tree.selectedItems()
        for item in items:
            idx = self.premise_tree.indexOfTopLevelItem(item)
            self.premise_tree.takeTopLevelItem(idx)

    def add_effect(self):
        dlg = EffectDialog(self, self.effect_dict)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            k, v = dlg.get_data()
            QTreeWidgetItem(self.effect_tree, [str(k), str(v)])

    def del_effect(self):
        items = self.effect_tree.selectedItems()
        for item in items:
            idx = self.effect_tree.indexOfTopLevelItem(item)
            self.effect_tree.takeTopLevelItem(idx)

    def apply_changes(self):
        if not self.current_uid:
            return
        data = self.config_data[self.current_uid]
        data["type"] = self.type_combo.currentData()
        data["related_knowledge"] = self.rk_spin.value()
        data["content_template"] = self.content_edit.toPlainText()
        
        premises = {}
        for i in range(self.premise_tree.topLevelItemCount()):
            item = self.premise_tree.topLevelItem(i)
            premises[item.text(0)] = int(item.text(1))
        data["premise_condition"] = premises
        
        effects = {}
        for i in range(self.effect_tree.topLevelItemCount()):
            item = self.effect_tree.topLevelItem(i)
            effects[item.text(0)] = int(item.text(1))
        data["action_effect"] = effects
        
        self.update_list()
        QMessageBox.information(self, "提示", "修改已应用。")

    def save_data(self):
        with open("default.json", "w", encoding="utf-8") as f:
            json.dump(self.config_data, f, ensure_ascii=False, indent=4)
        QMessageBox.information(self, "提示", "已保存到 default.json")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
