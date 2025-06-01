#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import csv
import os
import json
import platform
import datetime

config_dir = os.path.join("data", "csv")
os.system("cp ./tools/DieloliEventEditor/default.json ./data/event/")
event_dir = os.path.join("data", "event")
os.system("cp ./tools/DieloliAIEditor/default.json ./data/target/")
target_dir = os.path.join("data", "target")
os.system("cp ./tools/DieloliClubEditor/default.json ./data/club/")
club_dir = os.path.join("data", "club")
os.system("cp ./tools/ai_play/policy_model.pth ./data/")
clothing_dir = os.path.join("data", "clothing")
os.system("cp ./tools/DieloliClothingEditor/default.json ./data/clothing/")
config_data = {}
config_def_str = ""
config_po = ""
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
    print(i)
    build_csv_config(now_file, i)
    index += 1

event_file_list = os.listdir(event_dir)
event_list = []
for i in event_file_list:
    if i.split(".")[1] != "json":
        continue
    now_event_path = os.path.join(event_dir, i)
    with open(now_event_path, "r", encoding="utf-8") as event_file:
        now_event_data = json.loads(event_file.read())
        for event_id in now_event_data:
            now_event = now_event_data[event_id]
            event_list.append(now_event)
            now_event_text = now_event["text"]
            if now_event_text not in msgData:
                config_po += f"#: Event:{event_id}\n"
                config_po += f'msgid "{now_event_text}"\n'
                config_po += 'msgstr ""\n\n'
                msgData.add(now_event_text)
config_data["Event"] = {}
config_data["Event"]["data"] = event_list
config_data["Event"]["gettext"] = {}
config_data["Event"]["gettext"]["text"] = 1

target_file_list = os.listdir(target_dir)
target_list = []
for i in target_file_list:
    if i.split(".")[1] != "json":
        continue
    now_target_path = os.path.join(target_dir, i)
    with open(now_target_path, "r", encoding="utf-8") as target_file:
        now_target_data = json.loads(target_file.read())
        for target_id in now_target_data:
            now_target = now_target_data[target_id]
            target_list.append(now_target)
config_data["Target"] = {}
config_data["Target"]["data"] = target_list

club_file_list = os.listdir(club_dir)
club_list = []
for i in club_file_list:
    if i.split(".")[1] != "json":
        continue
    now_club_path = os.path.join(club_dir, i)
    with open(now_club_path, "r", encoding="utf-8") as club_file:
        now_club_data = json.loads(club_file.read())
        for club_id in now_club_data:
            now_club = now_club_data[club_id]
            club_list.append(now_club)
config_data["Club"] = {}
config_data["Club"]["data"] = club_list

clothing_file_list = os.listdir(clothing_dir)
clothing_list = []
clothing_suit_list = []
for i in clothing_file_list:
    if i.split(".")[1] != "json":
        continue
    now_clothing_path = os.path.join(clothing_dir, i)
    with open(now_clothing_path, "r", encoding="utf-8") as clothing_file:
        now_clothing_file_all_data = json.loads(clothing_file.read())
        for clothing_id in now_clothing_file_all_data["clothing"]:
            now_clothing = now_clothing_file_all_data["clothing"][clothing_id]
            clothing_list.append(now_clothing)
            clothing_name = now_clothing["name"]
            clothing_describe = now_clothing["describe"]
            if clothing_name not in msgData:
                config_po += f"#:  ClothingName:{clothing_id}"
                config_po += f'msgid "{clothing_name}"'
                config_po += 'msgstr ""\n\n'
                msgData.add(clothing_name)
            if clothing_describe not in msgData:
                config_po += f"#:  ClothingDescribe:{clothing_id}"
                config_po += f'msgid "{clothing_describe}"'
                config_po += 'msgstr ""\n\n'
                msgData.add(clothing_name)
        for suit_id in now_clothing_file_all_data["suit"]:
            now_suit = now_clothing_file_all_data["suit"][suit_id]
            clothing_suit_list.append(now_suit)
            suit_name = now_suit["name"]
            if suit_name not in msgData:
                config_po += f"#: ClothingSuitName:{suit_id}"
                config_po += f'msgid "{suit_name}"'
                config_po += 'msgstr ""\n\n'
                msgData.add(suit_name)

config_data["Clothing"] = {}
config_data["Clothing"]["data"] = clothing_list
config_data["Clothing"]["gettext"] = {}
config_data["Clothing"]["gettext"]["name"] = 1
config_data["Clothing"]["gettext"]["describe"] = 1
config_data["ClothingSuit"] = {}
config_data["ClothingSuit"]["data"] = clothing_suit_list
config_data["ClothingSuit"]["gettext"] = {}
config_data["ClothingSuit"]["gettext"]["describe"] = 1


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
    version = f"{now_time.year}.{now_time.month}.{now_time.day}.{now_time.hour}.{now_time.minute}"
    version_data = {"version": version}
    json.dump(version_data, package_file, ensure_ascii=0)

if platform.system() == "Linux":
    os.system("python3 ./buildpo.py")
    po_file_dir = os.path.join("data", "po")
    origin_po_path = os.path.join(po_file_dir, "zh_CN", "LC_MESSAGES", "dieloli.po")
    new_po_path = "new_po.po"
    with open(new_po_path, "w", encoding="utf-8") as po_file:
        po_file.write("# This file is automatically generated\n")
        po_file.write("#\n")
        po_file.write('msgid ""\n')
        po_file.write('msgstr ""\n')
        po_file.write('"Content-Type: text/plain; charset=UTF-8\\n"\n')
        po_file.write('"Language: zh_CN\\n"\n')
        po_file.write(config_po)
    os.system(f"msgcat {origin_po_path} {new_po_path} -o {origin_po_path}")
    os.remove(new_po_path)
    print("update origin po")
    english_po_path = os.path.join(po_file_dir, "en_US", "LC_MESSAGES", "dieloli.po")
    print("update English po")
    os.system(f"msgmerge --update --no-fuzzy-matching {english_po_path} {origin_po_path}")
    os.system("python3 ./buildmo.py")

print("Config Building End")
