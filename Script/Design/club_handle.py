import random
from Script.Config import game_config
from Script.Core import cache_control, game_type
from Script.Design import handle_premise

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


def init_club_data():
    """ 初始化社团数据 """
    club_max = random.randint(20, 30)
    club_config_max = len(game_config.config_club_data)
    club_max = min(club_max, club_config_max)
    club_id_set = set(game_config.config_club_data.keys())
    join_club_character = set()
    student_set = cache.student_character_set.copy()
    teacher_set = cache.teacher_character_set.copy()
    if 0 in student_set:
        student_set.remove(0)
    if 0 in teacher_set:
        teacher_set.remove(0)
    cache.club_create_judge = True
    while 1:
        if len(cache.all_club_data) + len(club_id_set) < club_max:
            club_max = len(cache.all_club_data) + len(club_id_set)
        if len(cache.all_club_data) >= club_max:
            break
        if not len(teacher_set):
            break
        if not len(student_set):
            break
        now_club_id = ""
        while 1:
            now_club_id = random.choice(list(club_id_set))
            if now_club_id not in cache.all_club_data:
                break
        club_id_set.remove(now_club_id)
        config_data = game_config.config_club_data[now_club_id]
        now_student_set = set()
        for student_id in student_set:
            now_judge = True
            for premise in config_data.premise_data:
                if not handle_premise.handle_premise(premise, student_id):
                    now_judge = False
                    break
            if now_judge:
                now_student_set.add(student_id)
        if not len(now_student_set):
            continue
        club_data = game_type.ClubData()
        club_data.uid = now_club_id
        club_data.name = config_data.name
        club_data.premise_data = config_data.premise_data
        club_data.theme = config_data.theme
        club_data.activity_list = config_data.activity_list
        teacher_id = random.choice(list(teacher_set))
        club_data.teacher = teacher_id
        teacher_data = cache.character_data[teacher_id]
        teacher_identity_data = game_type.ClubIdentity()
        teacher_identity_data.club_identity = 2
        teacher_identity_data.club_uid = now_club_id
        teacher_data.identity_data[teacher_identity_data.cid] = teacher_identity_data
        teacher_set.remove(teacher_id)
        student_index = random.randint(20, 40)
        student_index = min(len(now_student_set), student_index)
        member_list = random.sample(list(now_student_set), student_index)
        president_id = 0
        president_age = 0
        for member_id in member_list:
            club_data.character_set.add(member_id)
            student_set.remove(member_id)
            member_data = cache.character_data[member_id]
            if member_data.age > president_age:
                president_id = member_id
                president_age = member_data.age
            member_identity_data = game_type.ClubIdentity()
            member_identity_data.club_uid = now_club_id
            member_data.identity_data[member_identity_data.cid] = member_identity_data
        president_identity_data = game_type.ClubIdentity()
        president_identity_data.club_identity = 1
        president_identity_data.club_uid = now_club_id
        president_data = cache.character_data[president_id]
        president_data.identity_data[president_identity_data.cid] = president_identity_data
        club_data.president = member_id
        for activity in club_data.activity_list.values():
            for activity_time in activity.activity_time_list.values():
                week_day = activity_time.week_day
                start_time = (activity_time.start_hour, activity_time.start_minute)
                end_time = (activity_time.end_hour, activity_time.end_minute)
                club_data.activity_time_dict.setdefault(week_day, {})
                current_time = start_time
                while current_time != end_time:
                    hour, minute = current_time
                    club_data.activity_time_dict[week_day].setdefault(hour, {})
                    club_data.activity_time_dict[week_day][hour][minute] = activity_time.uid
                    minute += 1
                    if minute >= 60:
                        minute = 0
                        hour += 1
                    if hour >= 24:
                        break
                    current_time = (hour, minute)
        cache.all_club_data[club_data.uid] = club_data
    cache.club_create_judge = False

