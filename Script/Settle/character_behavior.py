from types import FunctionType
from Script.Design import (
    settle_behavior,
    map_handle,
    cooking,
    game_time,
    character,
    constant,
)
from Script.Core import (
    get_text,
    game_type,
    cache_control,
)
from Script.Config import game_config, normal_config
from Script.UI.Moudle import draw

_: FunctionType = get_text._
""" 翻译api """
window_width: int = normal_config.config_normal.text_width
""" 窗体宽度 """
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.MOVE_TO_TARGET_SCENE)
def handle_move_to_target_scene(
        character_id: int,
        add_time: int,
        change_data: game_type.CharacterStatusChange,
        now_time: int,
):
    """
    移动至目标场景
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    if character_data.behavior.move_target:
        old_scene_path = map_handle.get_map_system_path_str_for_list(character_data.position)
        old_scene_data: game_type.Scene = cache.scene_data[old_scene_path]
        for now_character in old_scene_data.character_list:
            if now_character == character_id:
                continue
            now_character_data: game_type.Character = cache.character_data[now_character]
            if now_character_data.target_character_id == character_id:
                now_character_data.target_character_id = 0
        map_handle.character_move_scene(
            character_data.position, character_data.behavior.move_target, character_id
        )


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.EAT_FOOD)
def handle_eat_food(
        character_id: int,
        add_time: int,
        change_data: game_type.CharacterStatusChange,
        now_time: int,
):
    """
    食用指定食物
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    if character_data.behavior.eat_food is not None:
        food: game_type.Food = character_data.behavior.eat_food
        eat_weight = 100
        if food.weight < eat_weight:
            eat_weight = food.weight
        new_food = cooking.separate_weight_food(food, eat_weight)
        for feel in new_food.feel:
            now_feel_value = new_food.feel[feel]
            character_data.status.setdefault(feel, 0)
            change_data.status.setdefault(feel, 0)
            if feel in {27, 28}:
                character_data.day_add_calories += now_feel_value
                now_feel_value /= 100
                character_data.status[feel] -= now_feel_value
                if character_data.status[feel] < 0:
                    character_data.status[feel] = 0
                change_data.status[feel] -= now_feel_value
            else:
                character_data.status[feel] += now_feel_value
                change_data.status[feel] += now_feel_value
        food_name = ""
        if food.recipe == -1:
            food_config = game_config.config_food[food.id]
            food_name = food_config.name
        else:
            food_name = cache.recipe_data[food.recipe].name
        if food.weight <= 0:
            if food.uid in character_data.food_bag:
                del character_data.food_bag[food.uid]
        character_data.behavior.food_name = food_name
        character_data.behavior.food_quality = food.quality


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.INTERRUPT_TARGET_ACTIVITY)
def handle_interrupt_target_activity(
        character_id: int,
        add_time: int,
        change_data: game_type.CharacterStatusChange,
        now_time: int,
):
    """
    打断交互目标活动
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    if target_data.state == constant.CharacterStatus.STATUS_DEAD:
        return
    if (
            target_data.behavior.behavior_id
            and target_data.behavior.start_time <= character_data.behavior.start_time
    ):
        target_end_time = game_time.get_sub_date(
            target_data.behavior.duration, old_date=target_data.behavior.start_time
        )
        old_statue = target_data.state
        target_data.behavior = game_type.Behavior()
        character.init_character_behavior_start_time(
            target_data.cid, character_data.behavior.start_time
        )
        target_data.state = constant.CharacterStatus.STATUS_ARDER
        if target_end_time >= character_data.behavior.start_time:
            if not character_id or not character_data.target_character_id:
                name_draw = draw.NormalDraw()
                name_draw.text = (
                        "\n"
                        + target_data.name
                        + _("停止了")
                        + game_config.config_status[old_statue].name
                )
                name_draw.width = window_width
                name_draw.draw()
                line_feed = draw.NormalDraw()
                line_feed.text = "\n"
                line_feed.draw()


@settle_behavior.add_settle_behavior_effect(
    constant.BehaviorEffect.ADD_SMALL_ATTEND_CLASS_EXPERIENCE
)
def handle_add_small_attend_class_experience(
        character_id: int,
        add_time: int,
        change_data: game_type.CharacterStatusChange,
        now_time: int,
):
    """
    按学习课程增加少量对应技能经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    course = character_data.behavior.course_id
    if course in game_config.config_course_knowledge_experience_data:
        knowledge_experience_data = game_config.config_course_knowledge_experience_data[course]
        for knowledge in knowledge_experience_data:
            knowledge_interest = character_data.knowledge_interest[knowledge]
            experience = knowledge_experience_data[knowledge] / 45 * add_time * knowledge_interest
            character_data.knowledge.setdefault(knowledge, 0)
            character_data.knowledge[knowledge] += experience
            change_data.knowledge.setdefault(knowledge, 0)
            change_data.knowledge[knowledge] += experience
    if course in game_config.config_course_language_experience_data:
        language_experience_data = game_config.config_course_language_experience_data[course]
        for language in language_experience_data:
            language_interest = character_data.language_interest[language]
            experience = language_experience_data[language] / 45 * add_time * language_interest
            character_data.language.setdefault(language, 0)
            character_data.language[language] += experience
            change_data.language.setdefault(language, 0)
            change_data.language[language] += experience


@settle_behavior.add_settle_behavior_effect(
    constant.BehaviorEffect.ADD_STUDENTS_COURSE_EXPERIENCE_FOR_IN_CLASSROOM
)
def handle_add_student_course_experience_for_in_class_room(
        character_id: int,
        add_time: int,
        change_data: game_type.CharacterStatusChange,
        now_time: int,
):
    """
    按课程增加教室内本班级学生的技能经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    scene_data: game_type.Scene = cache.scene_data[character_data.classroom]
    course = character_data.behavior.course_id
    for now_character in (
            scene_data.character_list & cache.classroom_students_data[character_data.classroom]
    ):
        now_character_data: game_type.Character = cache.character_data[now_character]
        if course in game_config.config_course_knowledge_experience_data:
            knowledge_experience_data = game_config.config_course_knowledge_experience_data[course]
            for knowledge in knowledge_experience_data:
                character_data.knowledge.setdefault(knowledge, 0)
                experience = character_data.knowledge[knowledge] / 1000
                knowledge_interest = now_character_data.knowledge_interest[knowledge]
                experience *= knowledge_interest
                now_character_data.knowledge.setdefault(knowledge, 0)
                now_character_data.knowledge[knowledge] += experience
        if course in game_config.config_course_language_experience_data:
            language_experience_data = game_config.config_course_language_experience_data[course]
            for language in language_experience_data:
                language_interest = character_data.language_interest[language]
                character_data.language.setdefault(language, 0)
                experience = character_data.language[language] / 1000 * language_interest
                now_character_data.language.setdefault(language, 0)
                now_character_data.language[language] += experience
