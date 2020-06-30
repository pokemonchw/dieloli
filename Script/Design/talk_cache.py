from Script.Core.game_type import Character,Food

me: Character = None
""" 玩家自己 """
tg: Character = None
""" 玩家当前选中的对话目标 """
scene:str = ""
""" 当前场景名字 """
scene_tag:str = ""
""" 当前场景标签 """
tg_eat:str = ""
""" 玩家当前选中的对话目标食用的食物的名字 """
tg_food:Food = None
""" 玩家当前选中的对话目标食用的食物的数据 """
