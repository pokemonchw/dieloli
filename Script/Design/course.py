import math
import random
from Script.Core import text_loading, value_handle, cache_contorl, constant
from Script.Core.game_type import Character


def init_phase_course_hour():
    """
    初始化各班级课时
    """
    phase_course_time = text_loading.get_text_data(
        constant.FilePath.PHASE_COURSE_PATH, "CourseTime"
    )
    primary_weight = text_loading.get_text_data(
        constant.FilePath.PHASE_COURSE_PATH, "PrimarySchool"
    )
    junior_middle_weight = text_loading.get_text_data(
        constant.FilePath.PHASE_COURSE_PATH, "JuniorMiddleSchool"
    )
    senior_high_weight = text_loading.get_text_data(
        constant.FilePath.PHASE_COURSE_PATH, "SeniorHighSchool"
    )
    now_weight_list = (
        primary_weight + junior_middle_weight + senior_high_weight
    )
    all_class_hour_data = {}
    phase_index = 0
    for phase in now_weight_list:
        phase_weight_regin = value_handle.get_region_list(phase)
        weight_max = 0
        weight_max = sum(map(int, phase_weight_regin.keys()))
        class_hour_data = {}
        class_hour_max = 0
        if phase_index <= 5:
            class_hour_max = phase_course_time["PrimarySchool"]
        elif phase_index <= 8:
            class_hour_max = phase_course_time["JuniorMiddleSchool"]
        else:
            class_hour_max = phase_course_time["SeniorHighSchool"]
        class_hour_data = {
            phase_weight_regin[region]: math.ceil(
                class_hour_max * (int(region) / weight_max)
            )
            for region in phase_weight_regin
        }
        now_class_hour_max = sum(class_hour_data.values())
        while now_class_hour_max != class_hour_max:
            for course in class_hour_data:
                if now_class_hour_max == class_hour_max:
                    break
                elif (
                    class_hour_data[course] > 1
                    and now_class_hour_max > class_hour_max
                ):
                    class_hour_data[course] -= 1
                    now_class_hour_max -= 1
                elif now_class_hour_max < class_hour_max:
                    class_hour_data[course] += 1
                    now_class_hour_max += 1
        more_hour = 0
        while 1:
            for course in class_hour_data:
                if more_hour > 0 and class_hour_data[course] < 14:
                    class_hour_data[course] += 1
                    more_hour -= 1
                elif more_hour > 0 and class_hour_data[course] > 14:
                    more_hour += class_hour_data[course] - 14
                    class_hour_data[course] -= class_hour_data[course] - 14
            if more_hour == 0:
                break
        all_class_hour_data[phase_index] = class_hour_data
        phase_index += 1
    cache_contorl.course_data["ClassHour"] = all_class_hour_data
    init_phase_course_hour_experience()


def init_class_time_table():
    """
    初始化各班级课程表
    """
    course_session = text_loading.get_game_data(
        constant.FilePath.COURSE_SESSION_PATH
    )
    class_time_table = {}
    for phase in cache_contorl.course_data["ClassHour"]:
        class_time = {}
        class_time_table[phase] = {}
        class_day = 0
        if phase <= 5:
            class_time = course_session["PrimarySchool"]
            class_day = 6
        elif phase <= 7:
            class_time = course_session["JuniorMiddleSchool"]
            class_day = 7
        else:
            class_time = course_session["SeniorHighSchool"]
            class_day = 8
        class_hour = cache_contorl.course_data["ClassHour"][phase]
        class_hour_index = {}
        for course in reversed(list(class_hour.keys())):
            class_hour_index.setdefault(course, 0)
            while class_hour_index[course] < class_hour[course]:
                for day in range(1, class_day):
                    old_day = day - 1
                    if old_day < 0:
                        old_day = class_day - 1
                    class_time_table[phase].setdefault(day, {})
                    class_time_table[phase].setdefault(old_day, {})
                    for i in range(1, len(class_time)):
                        if (
                            i not in class_time_table[phase][old_day]
                            and i not in class_time_table[phase][day]
                        ):
                            class_time_table[phase][day][i] = course
                            class_hour_index[course] += 1
                            break
                        elif i not in class_time_table[phase][day]:
                            if course != class_time_table[phase][old_day][i]:
                                class_time_table[phase][day][i] = course
                                class_hour_index[course] += 1
                                break
                            elif i == len(class_time) - 1:
                                class_time_table[phase][day][i] = course
                                class_hour_index[course] += 1
                                break
                            elif all(
                                [
                                    k in class_time_table[phase][day]
                                    for k in range(len(class_time[i + 1 :]))
                                ]
                            ):
                                class_time_table[phase][day][i] = course
                                class_hour_index[course] += 1
                                break
                    if class_hour_index[course] >= class_hour[course]:
                        break
    cache_contorl.course_data["ClassTimeTable"] = class_time_table


def init_class_teacher():
    """
    初始化各班级任课老师
    """
    teacher_index = len(
        cache_contorl.teacher_course_experience[
            list(cache_contorl.teacher_course_experience.keys())[0]
        ].keys()
    )
    course_max_a = 0
    course_max_b = 0
    vice_course_index_b = 0
    cache_contorl.course_data["ClassTeacher"] = {}
    for phase in cache_contorl.course_data["ClassHour"]:
        course_max_a += (
            len(cache_contorl.course_data["ClassHour"][phase].keys()) * 3
        )
        for course in cache_contorl.course_data["ClassHour"][phase]:
            if cache_contorl.course_data["ClassHour"][phase][course] > 7:
                course_max_b += 3
            else:
                course_max_b += 1
                vice_course_index_b += 1.5
    if teacher_index >= course_max_a:
        course_distribution_a()
    elif teacher_index >= course_max_b:
        course_distribution_b()


def course_abmain_distribution():
    """
    课时分配流程AB通用主课时分配流程
    """
    for phase in range(12, 0, -1):
        class_list = cache_contorl.place_data["Classroom_" + str(phase)]
        cache_contorl.course_data["ClassTeacher"][
            "Classroom_" + str(phase)
        ] = {}
        for classroom in class_list:
            cache_contorl.course_data["ClassTeacher"][
                "Classroom_" + str(phase)
            ][classroom] = {}
            for course in cache_contorl.course_data["ClassHour"][phase - 1]:
                if (
                    cache_contorl.course_data["ClassHour"][phase - 1][course]
                    > 7
                ):
                    cache_contorl.course_data["ClassTeacher"][
                        "Classroom_" + str(phase)
                    ][classroom][course] = []
                    for teacher in cache_contorl.teacher_course_experience[
                        course
                    ]:
                        if teacher not in teacher_data:
                            teacher_data[teacher] = 0
                            cache_contorl.course_data["ClassTeacher"][
                                "Classroom_" + str(phase)
                            ][classroom][course].append(teacher)
                            break


teacher_data = {}


def course_distribution_a():
    """
    课时分配流程A
    """
    course_abmain_distribution()
    for phase in range(1, 13):
        class_list = cache_contorl.place_data["Classroom_" + str(phase)]
        cache_contorl.course_data["ClassTeacher"][
            "Classroom_" + str(phase)
        ] = {}
        for classroom in class_list:
            cache_contorl.course_data["ClassTeacher"][
                "Classroom_" + str(phase)
            ][classroom] = {}
            for course in cache_contorl.course_data["ClassHour"][phase - 1]:
                if (
                    cache_contorl.course_data["ClassHour"][phase - 1][course]
                    <= 7
                ):
                    cache_contorl.course_data["ClassTeacher"][
                        "Classroom_" + str(phase)
                    ][classroom][course] = []
                    for teacher in cache_contorl.teacher_course_experience[
                        course
                    ]:
                        if teacher not in teacher_data:
                            teacher_data[teacher] = 0
                            cache_contorl.course_data["ClassTeacher"][
                                "Classroom_" + str(phase)
                            ][classroom][course].append(teacher)
                            break


def course_distribution_b():
    """
    课时分配流程B
    """
    course_abmain_distribution()
    for phase in range(1, 13):
        class_list = cache_contorl.place_data["Classroom_" + str(phase)]
        cache_contorl.course_data["ClassTeacher"][
            "CLassroom_" + str(phase)
        ] = {}
        teacher_course_index = 0
        for course in cache_contorl.course_data["ClassHour"][phase - 1]:
            for classroom in class_list:
                cache_contorl.course_data["ClassTeacher"][
                    "Classroom_" + str(phase)
                ][classroom] = {}
                if (
                    cache_contorl.course_data["ClassHour"][phase - 1][course]
                    <= 7
                ):
                    cache_contorl.course_data["ClassTeacher"][
                        "Classroom_" + str(phase)
                    ][classroom][course] = []
                    for teacher in cache_contorl.teacher_course_experience[
                        course
                    ]:
                        if teacher not in teacher_data:
                            cache_contorl.course_data["ClassTeacher"][
                                "Classroom_" + str(phase)
                            ][classroom][course].append(teacher)
                            teacher_course_index += 1
                            if teacher_course_index == 2:
                                teacher_course_index = 0
                                teacher_data[teacher] = 0
                            break


course_knowledge_data = text_loading.get_game_data(
    constant.FilePath.COURSE_PATH
)


def init_phase_course_hour_experience():
    """
    按年级计算各科目课时经验标准量
    """
    phase_experience = {}
    for phase in cache_contorl.course_data["ClassHour"]:
        phase_experience[phase] = {}
        for course in cache_contorl.course_data["ClassHour"][phase]:
            course_hour = cache_contorl.course_data["ClassHour"][phase][course]
            for knowledge in course_knowledge_data[course]["Knowledge"]:
                if knowledge not in phase_experience[phase]:
                    phase_experience[phase][knowledge] = {}
                for skill in course_knowledge_data[course]["Knowledge"][
                    knowledge
                ]:
                    skill_experience = (
                        course_knowledge_data[course]["Knowledge"][knowledge][
                            skill
                        ]
                        * course_hour
                        * 38
                    )
                    if skill in phase_experience[phase][knowledge]:
                        phase_experience[phase][knowledge][
                            skill
                        ] += skill_experience
                    else:
                        phase_experience[phase][knowledge][
                            skill
                        ] = skill_experience
    cache_contorl.course_data["PhaseExperience"] = phase_experience


def init_character_knowledge():
    """
    初始化所有角色知识等级
    """
    for i in cache_contorl.character_data:
        character = cache_contorl.character_data[i]
        character_age = character.age
        class_grade = 11
        if character_age <= 18 and character_age >= 7:
            class_grade = character_age - 7
        init_experience_for_grade(class_grade, character)
        cache_contorl.character_data[i] = character
        if character_age > 18:
            init_teacher_knowledge(character)
            for course in course_knowledge_data:
                if course not in cache_contorl.teacher_course_experience:
                    cache_contorl.teacher_course_experience.setdefault(
                        course, {}
                    )
                now_course_experience = 0
                for knowledge in course_knowledge_data[course]["Knowledge"]:
                    for skill in course_knowledge_data[course]["Knowledge"][
                        knowledge
                    ]:
                        if knowledge == "Language":
                            now_course_experience += character.language[skill]
                        else:
                            now_course_experience += character.knowledge[
                                knowledge
                            ][skill]
                cache_contorl.teacher_course_experience[course][
                    i
                ] = now_course_experience


def init_teacher_knowledge(character: Character) -> dict:
    """
    按年龄修正教师知识等级
    Keyword arguments:
    character = 角色对象
    """
    study_year = character.age - 18
    for knowledge in character.knowledge:
        for skill in character.knowledge[knowledge]:
            character.knowledge[knowledge][skill] += (
                character.knowledge[knowledge][skill]
                / 12
                * study_year
                * random.uniform(0.25, 0.75)
            )
    for language in character.language:
        character.knowledge[knowledge][skill] += (
            character.language[language]
            / 12
            * study_year
            * random.uniform(0.25, 0.75)
        )


course_data = text_loading.get_game_data(constant.FilePath.COURSE_PATH)


def init_experience_for_grade(class_grade: str, character: Character):
    """
    按年级生成角色初始经验数据
    """
    phase_experience_data = cache_contorl.course_data["PhaseExperience"]
    for garde in range(class_grade):
        experience_data = phase_experience_data[garde]
        for knowledge in experience_data:
            if knowledge == "Language":
                for skill in experience_data[knowledge]:
                    skill_experience = experience_data[knowledge][skill]
                    skill_interest = character.interest[skill]
                    skill_experience *= skill_interest
                    if skill in character.language:
                        character.language[skill] += skill_experience
                    else:
                        character.language[skill] = skill_experience
            else:
                if knowledge not in character.knowledge:
                    character.knowledge.setdefault(knowledge, {})
                for skill in experience_data[knowledge]:
                    skill_experience = experience_data[knowledge][skill]
                    skill_interest = character.interest[skill]
                    skill_experience *= skill_interest
                    if skill in character.knowledge[knowledge]:
                        character.knowledge[knowledge][
                            skill
                        ] += skill_experience
                    else:
                        character.knowledge[knowledge][
                            skill
                        ] = skill_experience


def calculation_character_learning_experience(
    character_id: int, skill: str, elapsed_time: int
):
    """
    计算角色花费指定时间学习指定技能的结果
    Keyword arguments:
    character_id -- 角色id
    skill -- 技能id
    elapsed_time -- 经过时间(单位分钟)
    """
