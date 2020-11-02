import timeit
from functools import wraps
from Script.Core import text_loading, era_print, constant, cache_contorl
from Script.Design import game_time, update, character
from Script.Flow import buy_food, eat_food


handle_instruct_data = {}
""" 指令处理数据 """


def handle_instruct(instruct: str):
    """
    处理执行指令
    Keyword arguments:
    instruct -- 指令id
    """
    if instruct in handle_instruct_data:
        handle_instruct_data[instruct]()


def add_instruct(instruct: str):
    """
    添加指令处理
    Keyword arguments:
    instruct -- 指令id
    """

    def decorator(func):
        @wraps(func)
        def return_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        handle_instruct_data[instruct] = return_wrapper
        return return_wrapper

    return decorator


def handle_unknown_instruct():
    """
    处理未定义指令
    """
    era_print.line_feed_print(
        text_loading.get_text_data(constant.FilePath.MESSAGE_PATH, "42")
    )


@add_instruct("Rest")
def handle_rest():
    """
    处理休息指令
    """
    character.init_character_behavior_start_time(0)
    character_data = cache_contorl.character_data[0]
    character_data.behavior["Duration"] = 10
    character_data.behavior["BehaviorId"] = constant.Behavior.REST
    character_data.state = constant.CharacterStatus.STATUS_REST
    if character_data.hit_point > character_data.hit_point_max:
        character_data.hit_point = character_data.hit_point_max
    target_character = cache_contorl.character_data[character_data.target_character_id]
    if (
        target_character.state == constant.CharacterStatus.STATUS_ARDER
        and target_character.behavior["BehaviorId"] == constant.Behavior.SHARE_BLANKLY
    ):
        target_character.state = constant.CharacterStatus.STATUS_REST
        character.init_character_behavior_start_time(character_data.target_character_id)
        target_character.behavior["Duration"] = 10
        target_character.behavior["BehaviorId"] = constant.Behavior.REST
    update.game_update_flow(10)


@add_instruct("BuyFood")
def handle_buy_food():
    """
    处理购买食物指令
    """
    buy_food.buy_food()


@add_instruct("Eat")
def handle_eat():
    """
    处理进食指令
    """
    character.init_character_behavior_start_time(0)
    judge, now_food = eat_food.eat_food()
    if judge:
        character_data = cache_contorl.character_data[0]
        character_data.behavior["BehaviorId"] = constant.Behavior.EAT
        character_data.behavior["EatFood"] = now_food
        character_data.behavior["Duration"] = 1
        character_data.state = constant.CharacterStatus.STATUS_EAT
    update.game_update_flow(1)
