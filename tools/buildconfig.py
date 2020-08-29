import csv
import os
import json

config_dir = os.path.join("..","data","csv")
config_data = {}
config_def_str = ""
config_po = ""

def build_csv_config(file_path:str,file_name:str):
    with open(file_path,encoding="utf-8") as now_file:
        now_read = csv.DictReader(now_file)
        now_docstring_data = {}
        now_type_data = {}
        get_text_data = {}
        file_id = file_name.split(".")[0]
        i = 0
        config_data[file_id] = {
            "data":[],
            "gettext":{}
        }
        class_text = ""
        for row in now_read:
            if not i:
                for k in row:
                    now_docstring_data[k] = row[k]
                i += 1
                continue
            elif i == 1:
                for k in row:
                    now_type_data[k] = row[k]
                i += 1
                continue
            elif i == 2:
                for k in row:
                    get_text_data[k] = int(row[k])
                i += 1
                continue
            elif i == 3:
                class_text = list(row.values())[0]
                i += 1
                continue
            for k in now_type_data:
                now_type = now_type_data[k]
                if not len(row[k]):
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
                    build_config_po(row[k])
            config_data[file_id]["data"].append(row)
        config_data[file_id]["gettext"] = get_text_data
        build_config_def(file_id,now_type_data,now_docstring_data,class_text)


def build_config_def(class_name:str,value_type:dict,docstring:dict,class_text:str):
    global config_def_str
    config_def_str += ""
    config_def_str += "class " + class_name + ":"
    config_def_str += '\n    """ ' + class_text + ' """\n'
    for k in value_type:
        config_def_str += "\n    " + k + ": " + value_type[k] + "\n"
        config_def_str += "    " + '""" ' + docstring[k] + ' """'


def build_config_po(message:str):
    global config_po
    config_po += f'msgid "{message}"\n'
    config_po += f'msgstr "{message}"\n\n'


file_list = os.listdir(config_dir)
index = 0
for i in file_list:
    if i.split(".")[1] != "csv":
        continue
    if index:
        config_def_str += "\n\n\n"
    now_file = os.path.join(config_dir,i)
    build_csv_config(now_file,i)
    index += 1

po_path = os.path.join("..","data","po","zh_CN","zh_CN.po")

with open(po_path,"w") as po_file:
    po_file.write(config_po)

config_path = os.path.join("..","Script","Config","config_def.py")
config_def_str += '\n'
with open(config_path,"w") as config_file:
    config_file.write(config_def_str)

config_data_path = os.path.join("..","data","data.json")
with open(config_data_path,"w") as config_data_file:
    json.dump(config_data,config_data_file,ensure_ascii=0)
