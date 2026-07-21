import csv
import json
import uuid

# 规划整理的课程阶段性经验映射表
# 格式: course_id: (course_name, primary_skills, middle_skills, high_skills)
courses_map = {
    0: ("语文", 
        ["chinese", "literature", "morality", "ceremony"], 
        ["chinese", "literature", "ancient_chinese", "poetry", "ethic", "eloquence", "write"], 
        ["chinese", "literature", "ancient_chinese", "old_sinitic", "poetry", "ethic", "religion", "linguistics", "eloquence", "write"]
    ),
    1: ("数学", 
        ["mathematics"], 
        ["mathematics", "logic"], # wait logic is not a skill, let's omit logic
        ["mathematics", "numerology", "cryptography"]
    ),
    2: ("英语", 
        ["english"], 
        ["english", "linguistics"], 
        ["english", "linguistics", "french", "spanish"]
    ),
    3: ("信息技术", 
        ["computer"], 
        ["computer", "programming", "electronics"], 
        ["computer", "computer_science", "programming", "hacker", "electronics", "cryptography"]
    ),
    4: ("美术", 
        ["art", "painting"], 
        ["art", "painting", "shoot"], 
        ["art", "painting", "shoot", "manufacture"]
    ),
    5: ("音乐", 
        ["music", "singing"], 
        ["music", "singing", "play_music"], 
        ["music", "singing", "play_music", "write_music"]
    ),
    6: ("体育", 
        ["motion"], 
        ["motion", "first_aid", "tactics"], 
        ["motion", "first_aid", "tactics", "anatomy"]
    ),
    7: ("思想品德", 
        ["morality", "ceremony"], 
        ["morality", "ethic", "ceremony", "law_science"], 
        ["morality", "ethic", "ceremony", "law_science", "faith"]
    ),
    8: ("健康教育", 
        ["first_aid"], 
        ["first_aid", "anatomy", "biology"], 
        ["first_aid", "anatomy", "biology", "pharmacy"]
    ),
    9: ("自然", 
        ["ecology", "zoology", "botany"], 
        ["ecology", "zoology", "botany", "geography"], 
        ["ecology", "zoology", "botany", "geography", "geology", "meteorology", "entomology"]
    ),
    10: ("社会", 
        ["ceremony", "morality"], 
        ["history", "geography", "ceremony", "transaction"], 
        ["history", "geography", "law_science", "transaction", "sociology"] # wait sociology? no
    ),
    11: ("历史", 
        ["history"], 
        ["history", "ancient_chinese"], 
        ["history", "ancient_chinese", "old_sinitic", "religion"]
    ),
    12: ("生物", 
        ["biology", "botany", "zoology"], 
        ["biology", "botany", "zoology", "anatomy", "entomology"], 
        ["biology", "botany", "zoology", "anatomy", "microbiology", "virology", "becteriology", "mycology", "entomology"]
    ),
    13: ("地理", 
        ["geography"], 
        ["geography", "meteorology", "geology"], 
        ["geography", "meteorology", "geology", "astronomy", "astrology"]
    ),
    14: ("物理", 
        ["physics"], 
        ["physics", "mechanics"], 
        ["physics", "mechanics", "astronomy", "electronics"]
    ),
    15: ("化学", 
        ["chemistry"], 
        ["chemistry", "biology", "pharmacy"], 
        ["chemistry", "biology", "pharmacy", "apothecary"]
    ),
    16: ("舞蹈", 
        ["dance", "motion"], 
        ["dance", "motion", "performance", "art"], 
        ["dance", "motion", "performance", "art", "ceremony"]
    ),
    17: ("游泳", 
        ["swimming", "motion"], 
        ["swimming", "motion", "first_aid"], 
        ["swimming", "motion", "first_aid", "anatomy"]
    ),
}

# clean up sociology / logic since they don't exist
courses_map[1] = ("数学", ["mathematics"], ["mathematics"], ["mathematics", "numerology", "cryptography"])
courses_map[10] = ("社会", ["ceremony", "morality"], ["history", "geography", "ceremony", "transaction"], ["history", "geography", "law_science", "transaction", "politics"])
# wait politics? No. Let's just use what's available
courses_map[10] = ("社会", ["ceremony", "morality"], ["history", "geography", "ceremony", "transaction"], ["history", "geography", "law_science", "transaction", "eloquence"])

school_names = {
    0: "小学",
    1: "初中",
    2: "高中",
}

phase_names = {
    0: "一年级",
    1: "二年级",
    2: "三年级",
    3: "四年级",
    4: "五年级",
    5: "六年级",
}

exp_prefix_map = {
    0: "add_small_{}_experience",
    1: "add_medium_{}_experience",
    2: "add_large_{}_experience",
}

books = {}
seen = set()

with open("../../data/csv/SchoolPhaseCourse.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 4:
            try:
                school = int(row[1])
                phase = int(row[2])
                course = int(row[3])
            except ValueError:
                continue
            
            key = (school, phase, course)
            if key in seen:
                continue
            seen.add(key)
            
            s_name = school_names.get(school, f"学校{school}")
            p_name = phase_names.get(phase, f"{phase}年级")
            
            c_info = courses_map.get(course)
            if c_info:
                c_name = c_info[0]
                
                # Pick skills based on school
                if school == 0:
                    base_skills = c_info[1]
                elif school == 1:
                    base_skills = c_info[2]
                else:
                    base_skills = c_info[3]
                
                prefix_template = exp_prefix_map.get(school, "add_medium_{}_experience")
                
                # Make a textbook
                book_name = f"{s_name}{p_name}{c_name}教科书"
                settles = [prefix_template.format(skill) for skill in base_skills]
                
                uid = str(uuid.uuid4())
                
                # Determine book type
                if school == 0:
                    book_type = phase
                elif school == 1:
                    book_type = 6 + phase
                elif school == 2:
                    book_type = 9 + phase
                else:
                    book_type = 14
                
                books[uid] = {
                    "uid": uid,
                    "name": book_name,
                    "info": f"这是一本《{book_name}》，适合对应阶段的学生阅读和学习，涵盖了该学科的基础知识和核心概念。用心阅读可以获得相关属性的成长和经验。",
                    "settle_list": settles,
                    "type": book_type
                }

with open("default.json", "w", encoding="utf-8") as f:
    json.dump(books, f, ensure_ascii=False, indent=2)

print(f"Generated {len(books)} books with progressive depth!")