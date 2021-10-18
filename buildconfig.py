#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import csv
import os
import json
import datetime

config_dir = os.path.join("data", "csv")
talk_dir = os.path.join("data", "talk")
target_dir = os.path.join("data", "target")
config_data = {}
config_def_str = ""
config_po = "\n"
msgData = set()
class_data = set()


def build_csv_config(file_path: str, file_name: str):
    with open(file_path, encoding="utf-8") as now_file:
        now_read = csv.DictReader(now_file)
        now_docstring_data = {}
        now_type_data = {}
        get_text_data = {}
        file_id = file_name.split(".")[0]
        i = 0
        class_text = ""
        type_text = file_id
        config_data.setdefault(type_text, {})
        config_data[type_text].setdefault("data", [])
        config_data[type_text].setdefault("gettext", {})
        for row in now_read:
            if not i:
                for k in row:
                    now_docstring_data[k] = row[k]
                i += 1
                continue
            if i == 1:
                for k in row:
                    now_type_data[k] = row[k]
                i += 1
                continue
            if i == 2:
                for k in row:
                    get_text_data[k] = int(row[k])
                i += 1
                continue
            if i == 3:
                class_text = list(row.values())[0]
                i += 1
                continue
            for k in now_type_data:
                now_type = now_type_data[k]
                if not row[k]:
                    del row[k]
                    continue
                if now_type == "int":
                    row[k] = int(row[k])
                elif now_type == "str":
                    row[k] = str(row[k])
                elif now_type == "bool":
                    row[k] = int(row[k])
                elif now_type == "float":
                    row[k] = float(row[k])
                if get_text_data[k]:
                    build_config_po(row[k], type_text, k, row["cid"])
            config_data[type_text]["data"].append(row)
        config_data[type_text]["gettext"] = get_text_data
        build_config_def(type_text, now_type_data, now_docstring_data, class_text)


def build_config_def(class_name: str, value_type: dict, docstring: dict, class_text: str):
    global config_def_str
    if class_name not in class_data:
        config_def_str += "class " + class_name + ":"
        config_def_str += '\n    """ ' + class_text + ' """\n'
        for k in value_type:
            config_def_str += "\n    " + k + ": " + value_type[k] + "\n"
            config_def_str += "    " + '""" ' + docstring[k] + ' """'
        class_data.add(class_name)


def build_config_po(message: str, message_class: str, message_type: str, message_id: str):
    global config_po
    if message not in msgData:
        config_po += f"#: class:{message_class} id:{message_id} type:{message_type}\n"
        config_po += f'msgid "{message}"\n'
        config_po += 'msgstr ""\n\n'
        msgData.add(message)


def build_scene_config(data_path):
    global config_po
    for i in os.listdir(data_path):
        now_path = os.path.join(data_path, i)
        if os.path.isfile(now_path):
            if i == "Scene.json":
                with open(now_path, "r", encoding="utf-8") as now_file:
                    scene_data = json.loads(now_file.read())
                    scene_name = scene_data["SceneName"]
                    if scene_name not in msgData:
                        config_po += f"#: Scene:{now_path}\n"
                        config_po += f'msgid "{scene_name}"\n'
                        config_po += 'msgstr ""\n\n'
                        msgData.add(scene_name)
            elif i == "Map.json":
                with open(now_path, "r", encoding="utf-8") as now_file:
                    map_data = json.loads(now_file.read())
                    map_name = map_data["MapName"]
                    if map_name not in msgData:
                        config_po += f"#: Map:{now_path}\n"
                        config_po += f'msgid "{map_name}"\n'
                        config_po += 'msgstr ""\n\n'
                        msgData.add(map_name)
        else:
            build_scene_config(now_path)


file_list = os.listdir(config_dir)
index = 0
for i in file_list:
    if i.split(".")[1] != "csv":
        continue
    if index:
        config_def_str += "\n\n\n"
    now_file = os.path.join(config_dir, i)
    build_csv_config(now_file, i)
    index += 1

talk_file_list = os.listdir(talk_dir)
talk_list = []
for i in talk_file_list:
    if i.split(".")[1] != "json":
        continue
    now_talk_path = os.path.join(talk_dir, i)
    with open(now_talk_path, "w", encoding="utf-8") as talk_file:
        now_talk_data = json.load(now_talk_path)
        for talk_id in now_talk_data:
            now_talk = now_talk_data[talk_id]
            talk_list.append(now_talk)
            now_talk_text = now_talk["text"]
            if now_talk_text not in msgData:
                config_po += f"#: Talk:{talk_id}\n"
                config_po += f'msgid "{now_talk_text}"\n'
                config_po += 'msgstr ""\n\n'
                msgData.add(now_talk_text)
config_data["Talk"] = {}
config_data["Talk"]["data"] = talk_list
config_data["Talk"]["gettext"] = {}
config_data["Talk"]["gettext"]["text"] = 1

target_file_list = os.listdir(target_dir)
target_list = []
for i in target_file_list:
    if i.split(".")[1] != "json":
        continue
    now_target_path = os.path.join(target_dir, i)
    with open(now_target_path, "w", encoding="utf-8") as target_file:
        now_target_data = json.load(now_target_path)
        for target_id in now_target_data:
            now_target = now_target_data[target_id]
            target_list.append(now_target)
config_data["Target"] = {}
config_data["Target"]["data"] = target_list


map_path = os.path.join("data", "map")
build_scene_config(map_path)

config_path = os.path.join("Script", "Config", "config_def.py")
config_def_str += "\n"
with open(config_path, "w", encoding="utf-8") as config_file:
    config_file.write(config_def_str)

config_data_path = os.path.join("data", "data.json")
with open(config_data_path, "w", encoding="utf-8") as config_data_file:
    json.dump(config_data, config_data_file, ensure_ascii=0)

package_path = os.path.join("package.json")
with open(package_path, "w", encoding="utf-8") as package_file:
    now_time = datetime.datetime.now()
    version = f"{now_time.year}.{now_time.month}.{now_time.day}.{now_time.hour}"
    version_data = {"version": version}
    json.dump(version_data, package_file, ensure_ascii=0)

print("Config Building End")
