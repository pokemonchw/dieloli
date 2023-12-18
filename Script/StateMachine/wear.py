from Script.Design import handle_state_machine, constant
from Script.Core import cache_control, game_type

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_state_machine.add_state_machine(constant.StateMachine.WEAR_CLEAN_UNDERWEAR)
def character_wear_clean_underwear(character_id: int):
    """
    角色穿着干净的上衣
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 1 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[1]:
            clothing_data: game_type.Clothing = character_data.clothing[1][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[1] = value_dict[now_value]
        character_data.ai_target = 0


@handle_state_machine.add_state_machine(constant.StateMachine.WEAR_CLEAN_UNDERPANTS)
def character_wear_clean_underpants(character_id: int):
    """
    角色穿着干净的内裤
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 7 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[7]:
            clothing_data: game_type.Clothing = character_data.clothing[7][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[7] = value_dict[now_value]
        character_data.ai_target = 0


@handle_state_machine.add_state_machine(constant.StateMachine.WEAR_CLEAN_BRA)
def character_wear_clean_bra(character_id: int):
    """
    角色穿着干净的胸罩
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 6 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[6]:
            clothing_data: game_type.Clothing = character_data.clothing[6][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[6] = value_dict[now_value]
        character_data.ai_target = 0


@handle_state_machine.add_state_machine(constant.StateMachine.WEAR_CLEAN_PANTS)
def character_wear_clean_pants(character_id: int):
    """
    角色穿着干净的裤子
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 2 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[2]:
            clothing_data: game_type.Clothing = character_data.clothing[2][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[2] = value_dict[now_value]
        character_data.ai_target = 0


@handle_state_machine.add_state_machine(constant.StateMachine.WEAR_CLEAN_SKIRT)
def character_wear_clean_skirt(character_id: int):
    """
    角色穿着干净的短裙
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 3 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[3]:
            clothing_data: game_type.Clothing = character_data.clothing[3][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[3] = value_dict[now_value]
        character_data.ai_target = 0


@handle_state_machine.add_state_machine(constant.StateMachine.WEAR_CLEAN_SHOES)
def character_wear_clean_shoes(character_id: int):
    """
    角色穿着干净的鞋子
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 4 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[4]:
            clothing_data: game_type.Clothing = character_data.clothing[4][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[4] = value_dict[now_value]
        character_data.ai_target = 0


@handle_state_machine.add_state_machine(constant.StateMachine.WEAR_CLEAN_SOCKS)
def character_wear_clean_socks(character_id: int):
    """
    角色穿着干净的袜子
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 5 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[5]:
            clothing_data: game_type.Clothing = character_data.clothing[5][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[5] = value_dict[now_value]
        character_data.ai_target = 0


@handle_state_machine.add_state_machine(constant.StateMachine.WEAR_CLEAN_COAT)
def character_wear_clean_coat(character_id: int):
    """
    角色穿着干净的外套
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 0 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[0]:
            clothing_data: game_type.Clothing = character_data.clothing[0][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[0] = value_dict[now_value]
        character_data.ai_target = 0


@handle_state_machine.add_state_machine(constant.StateMachine.UNDRESS_UNDERWEAR)
def character_undress_underwear(character_id: int):
    """
    角色脱掉上衣
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.put_on[1] = ""
    character_data.ai_target = 0


@handle_state_machine.add_state_machine(constant.StateMachine.UNDRESS_UNDERPANTS)
def character_undress_underpants(character_id: int):
    """
    角色脱掉内裤
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.put_on[7] = ""
    character_data.ai_target = 0


@handle_state_machine.add_state_machine(constant.StateMachine.UNDRESS_BRA)
def character_undress_bra(character_id: int):
    """
    角色脱掉胸罩
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.put_on[6] = ""
    character_data.ai_target = 0


@handle_state_machine.add_state_machine(constant.StateMachine.UNDRESS_PANTS)
def character_undress_pants(character_id: int):
    """
    角色脱掉裤子
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.put_on[2] = ""
    character_data.ai_target = 0


@handle_state_machine.add_state_machine(constant.StateMachine.UNDRESS_SKIRT)
def character_undress_skirt(character_id: int):
    """
    角色脱掉裙子
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.put_on[3] = ""
    character_data.ai_target = 0


@handle_state_machine.add_state_machine(constant.StateMachine.UNDRESS_SHOES)
def character_undress_shoes(character_id: int):
    """
    角色脱掉鞋子
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.put_on[4] = ""
    character_data.ai_target = 0


@handle_state_machine.add_state_machine(constant.StateMachine.UNDRESS_SOCKS)
def character_undress_socks(character_id: int):
    """
    角色脱掉袜子
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.put_on[5] = ""
    character_data.ai_target = 0


@handle_state_machine.add_state_machine(constant.StateMachine.UNDRESS_COAT)
def character_undress_coat(character_id: int):
    """
    角色脱掉外套
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.put_on[0] = ""
    character_data.ai_target = 0


@handle_state_machine.add_state_machine(constant.StateMachine.TARGET_UNDRESS)
def character_target_undress(character_id: int):
    """
    让交互对象脱掉衣服
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.behavior.behavior_id = constant.Behavior.TARGET_UNDRESS
    character_data.behavior.duration = 2
    character_data.state = constant.CharacterStatus.STATUS_TARGET_UNDRESS


@handle_state_machine.add_state_machine(constant.StateMachine.WEAR_CLOTHING)
def character_wear_clothing(character_id: int):
    """
    穿上衣服
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.behavior.behavior_id = constant.Behavior.WEAR
    character_data.behavior.duration = 2
    character_data.state = constant.CharacterStatus.STATUS_WEAR


@handle_state_machine.add_state_machine(constant.StateMachine.UNDRESS)
def character_undress(character_id: int):
    """
    脱掉衣服
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.behavior.behavior_id = constant.Behavior.UNDRESS
    character_data.behavior.duration = 2
    character_data.state = constant.CharacterStatus.STATUS_UNDRESS
