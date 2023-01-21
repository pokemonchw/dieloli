import os
import csv
import cache_control

premise_path = os.path.join("..","premise.csv")
state_machine_path = os.path.join("..","state_machine.csv")


def load_config():
    """载入配置文件"""
    with open(premise_path, encoding="utf-8") as now_file:
        now_read = csv.DictReader(now_file)
        for i in now_read:
            cache_control.premise_data[i["cid"]] = i["premise"]
            cache_control.premise_type_data.setdefault(i["premise_type"], set())
            cache_control.premise_type_data[i["premise_type"]].add(i["cid"])
    with open(state_machine_path, encoding="utf-8") as now_file:
        now_read = csv.DictReader(now_file)
        for i in now_read:
            cache_control.state_machine_data[i["cid"]] = i["state_machine"]
            cache_control.state_machine_type_data.setdefault(i["state_machine_type"],set())
            cache_control.state_machine_type_data[i["state_machine_type"]].add(i["cid"])
