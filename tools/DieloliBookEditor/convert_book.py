import csv
import json
import uuid
import os

books = {}
with open("../../data/csv/Book.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i < 3: # skip first 3 rows of header
            continue
        if len(row) >= 2:
            cid = row[0]
            name = row[1]
            uid = str(uuid.uuid4())
            books[uid] = {
                "uid": uid,
                "name": name,
                "info": f"这是一本《{name}》，适合对应阶段的学生阅读和学习，涵盖了该学科的基础知识和核心概念。用心阅读可以获得相关属性的成长和经验。",
                "settle_list": []
            }

with open("default.json", "w", encoding="utf-8") as f:
    json.dump(books, f, ensure_ascii=False, indent=2)

print(f"Converted {len(books)} books!")
