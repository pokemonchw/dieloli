import random
import sys
from functools import wraps
from Script.Talk import route
from Script.Core import cache_contorl


class TalkObject:
    """
    口上对象
    """
    def __init__(self,occupation: str,instruct: str,func: callable):
        """
        构造口上对象
        Keyword arguments:
        occupation -- 口上所属的职业
        instruct -- 口上对应的命令id
        func -- 生成口上的执行函数
        """
        self.occupation = occupation
        self.instruct = instruct
        self.func = func


def add_talk(occupation: str,instruct: str):
    """
    添加口上
    Keyword arguments:
    occupation -- 口上所属的职业
    instruct -- 口上对应的命令id
    """
    def decorator(func):
        @wraps(func)
        def return_wrapper(*args,**kwargs):
            return random.choice(func(*args,**kwargs))
        now_talk = TalkObject(occupation,instruct,return_wrapper)
        cache_contorl.talk_data.setdefault(occupation,{})
        cache_contorl.talk_data[occupation] = return_wrapper
        create_variavle(func.__name__,return_wrapper)
        return return_wrapper
    return decorator


def create_variavle(name: str,var: callable):
    """
    构建口上模组
    Keyword arguments:
    name -- 口上函数名字(内部)
    var -- 口上函数(内部)
    """
    this_module = sys.modules[__name__]
    setattr(this_module,name,var)


def handle_instruct_talk(instruct: str):
    """
    处理对话对话
    Keyword arguments:
    instruct -- 操作id
    """
    now_target_character = cache_contorl.character_data['character'][cache_contorl.character_data['character_id']]
    if now_target_character.occupation in route.route_data:
        pass
    else:
        pass
