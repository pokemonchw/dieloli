import os
import csv
import cache_control

club_theme_path = os.path.join("..","ClubTheme.csv")
club_activity_path = os.path.join("..", "ClubActivity.csv")
premise_path = os.path.join("..","premise.csv")
status_path = os.path.join("..", "Status.csv")



def load_config():
    """载入配置文件"""
    with open(club_theme_path, encoding="utf-8") as now_file:
        now_read = csv.DictReader(now_file)
        for i in now_read:
            cache_control.club_theme[int(i["cid"])] = i["name"]
            cache_control.club_theme_data[i["name"]] = int(i["cid"])
    with open(club_activity_path, encoding="utf-8") as now_file:
        now_read = csv.DictReader(now_file)
        for i in now_read:
            cache_control.club_activity[int(i["cid"])] = i["name"]
            cache_control.club_activity_data[i["name"]] = int(i["cid"])
    with open(premise_path, encoding="utf-8") as now_file:
        now_read = csv.DictReader(now_file)
        for i in now_read:
            cache_control.premise_data[i["cid"]] = i["premise"]
            cache_control.premise_type_data.setdefault(i["premise_type"], set())
            cache_control.premise_type_data[i["premise_type"]].add(i["cid"])
    with open(status_path, encoding="utf-8") as now_file:
        now_read = csv.DictReader(now_file)
        for i in now_read:
            cache_control.activity_list[int(i["cid"])] = i["status"]
            cache_control.activity_list_data[i["status"]] = int(i["cid"])
