from types import FunctionType
from Script.Core import cache_control, game_type, get_text
from Script.Design import update, constant, handle_instruct
from Script.Config import normal_config


cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
width: int = normal_config.config_normal.text_width
""" 屏幕宽度 """


@handle_instruct.add_instruct(
    constant.Instruct.MASTURBATION,
    constant.InstructType.SEX,
    _("手淫"),
    {
        constant.Premise.NO_WEAR_UNDERPANTS,
        constant.Premise.NO_WEAR_PANTS,
        constant.Premise.IS_NOT_ASEXUAL,
    },
)
def handle_masturbation():
    """处理手淫指令"""
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.MASTURBATION
    character_data.state = constant.CharacterStatus.STATUS_MASTURBATION
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.INVITE_SEX,
    constant.InstructType.SEX,
    _("邀请做爱"),
    {
        constant.Premise.HAVE_TARGET,
    }
)
def handle_missionary_position():
    """处理邀请做爱指令"""
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    constant.settle_behavior_effect_data[constant.BehaviorEffect.INTERRUPT_TARGET_ACTIVITY](0,1,None,cache.game_time)
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.behavior.start_time = character_data.behavior.start_time
    character_data.behavior.duration = 1
    character_data.behavior.behavior_id = constant.Behavior.INVITE_SEX
    character_data.state = constant.CharacterStatus.STATUS_INVITE_SEX
    update.game_update_flow(1)


@handle_instruct.add_instruct(
    constant.Instruct.TOUCH_CLITORIS,
    constant.InstructType.SEX,
    _("抚摸阴蒂"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.TARGET_IS_FUTA_OR_WOMAN,
    }
)
def handle_touch_clitoris():
    """ 处理抚摸阴蒂指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.TOUCH_CLITORIS
    character_data.state = constant.CharacterStatus.STATUS_TOUCH_CLITORIS
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.TOUCH_PENIS,
    constant.InstructType.SEX,
    _("抚摸阴茎"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.TARGET_IS_FUTA_OR_MAN,
    }
)
def handle_touch_penis():
    """ 处理抚摸阴茎指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.TOUCH_PENIS
    character_data.state = constant.CharacterStatus.STATUS_TOUCH_PENIS
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.TOUCH_ANUS,
    constant.InstructType.SEX,
    _("抚摸肛门"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
    }
)
def handle_touch_anus():
    """ 处理抚摸肛门指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.TOUCH_ANUS
    character_data.state = constant.CharacterStatus.STATUS_TOUCH_ANUS
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.TOUCH_BUTT,
    constant.InstructType.SEX,
    _("抚摸屁股"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
    }
)
def handle_touch_butt():
    """ 处理抚摸屁股指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.TOUCH_BUTT
    character_data.state = constant.CharacterStatus.STATUS_TOUCH_BUTT
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.TOUCH_BACK,
    constant.InstructType.SEX,
    _("摸背"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
    }
)
def handle_touch_back():
    """ 处理摸背指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.TOUCH_BACK
    character_data.state = constant.CharacterStatus.STATUS_TOUCH_BACK
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.PENIS_RUB_FACE,
    constant.InstructType.SEX,
    _("阴茎蹭脸"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.IS_FUTA_OR_MAN,
    }
)
def handle_penis_rub_face():
    """ 处理阴茎蹭脸指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.TOUCH_BACK
    character_data.state = constant.CharacterStatus.STATUS_TOUCH_BACK
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.FINGER_INSERTION_VAGINAL,
    constant.InstructType.SEX,
    _("手指插入阴道"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.TARGET_IS_FUTA_OR_WOMAN,
    }
)
def handle_finger_insertion_vaginal():
    """ 处理手指插入阴道指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.FINGER_INSERTION_VAGINAL
    character_data.state = constant.CharacterStatus.STATUS_FINGER_INSERTION_VAGINAL
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.PLAY_CLITORIS,
    constant.InstructType.SEX,
    _("玩弄阴蒂"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.TARGET_IS_FUTA_OR_WOMAN,
    }
)
def handle_play_clitoris():
    """ 处理玩弄阴蒂指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.PLAY_CLITORIS
    character_data.state = constant.CharacterStatus.STATUS_PLAY_CLITORIS
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.PLAY_PENIS,
    constant.InstructType.SEX,
    _("玩弄阴茎"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.TARGET_IS_FUTA_OR_MAN,
    }
)
def handle_play_penis():
    """ 处理玩弄阴茎指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.PLAY_PENIS
    character_data.state = constant.CharacterStatus.STATUS_PLAY_PENIS
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.FINGER_INSERTION_ANUS,
    constant.InstructType.SEX,
    _("手指插入肛门"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
    }
)
def handle_finger_insertion_anus():
    """ 处理手指插入肛门指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.FINGER_INSERTION_ANUS
    character_data.state = constant.CharacterStatus.STATUS_FINGER_INSERTION_ANUS
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.PLAY_NIPPLE,
    constant.InstructType.SEX,
    _("玩弄乳头"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
    }
)
def handle_play_nipple():
    """ 处理玩弄乳头指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.PLAY_NIPPLE
    character_data.state = constant.CharacterStatus.STATUS_PLAY_NIPPLE
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.LICK_VAGINAL,
    constant.InstructType.SEX,
    _("舔阴"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.TARGET_IS_FUTA_OR_WOMAN,
    }
)
def handle_lick_vaginal():
    """ 处理舔阴指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.LICK_VAGINAL
    character_data.state = constant.CharacterStatus.STATUS_LICK_VAGINAL
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.LICK_PENIS,
    constant.InstructType.SEX,
    _("舔阴茎"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.TARGET_IS_FUTA_OR_MAN,
    }
)
def handle_lick_penis():
    """ 处理舔阴茎指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.LICK_PENIS
    character_data.state = constant.CharacterStatus.STATUS_LICK_PENIS
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.LICK_ANUS,
    constant.InstructType.SEX,
    _("舔肛"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
    }
)
def handle_lick_anus():
    """ 处理舔肛指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.LICK_ANUS
    character_data.state = constant.CharacterStatus.STATUS_LICK_ANUS
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.LICK_NIPPLE,
    constant.InstructType.SEX,
    _("吸舔乳头"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
    }
)
def handle_lick_nipple():
    """ 处理吸舔乳头指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.LICK_NIPPLE
    character_data.state = constant.CharacterStatus.STATUS_LICK_NIPPLE
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.LICK_FEET,
    constant.InstructType.SEX,
    _("舔足"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
    }
)
def handle_lick_feet():
    """ 处理舔足指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.LICK_FEET
    character_data.state = constant.CharacterStatus.STATUS_LICK_FEET
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.LICK_BUTT,
    constant.InstructType.SEX,
    _("舔屁股"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
    }
)
def handle_lick_butt():
    """ 处理舔屁股指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.LICK_BUTT
    character_data.state = constant.CharacterStatus.STATUS_LICK_BUTT
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.LICK_EARS,
    constant.InstructType.SEX,
    _("舔耳朵"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
    }
)
def handle_lick_ears():
    """ 处理舔耳朵指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.LICK_EARS
    character_data.state = constant.CharacterStatus.STATUS_LICK_EARS
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.LICK_BODY,
    constant.InstructType.SEX,
    _("舔全身"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
    }
)
def handle_lick_body():
    """ 处理舔全身指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 5
    character_data.behavior.behavior_id = constant.Behavior.LICK_BODY
    character_data.state = constant.CharacterStatus.STATUS_LICK_BODY
    update.game_update_flow(5)


@handle_instruct.add_instruct(
    constant.Instruct.LICK_FACE,
    constant.InstructType.SEX,
    _("舔脸"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
    }
)
def handle_lick_face():
    """ 处理舔脸指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.LICK_FACE
    character_data.state = constant.CharacterStatus.STATUS_LICK_FACE
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.TARGET_MOUTH_AND_HAND_SEX,
    constant.InstructType.SEX,
    _("让对方手交口交"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.IS_FUTA_OR_MAN,
    }
)
def handle_target_mouth_and_hand_sex():
    """ 处理让对方手交口交指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.TARGET_MOUTH_AND_HAND_SEX
    character_data.state = constant.CharacterStatus.STATUS_TARGET_MOUTH_AND_HAND_SEX
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.TARGET_MOUTH_AND_CHEST_SEX,
    constant.InstructType.SEX,
    _("让对方乳交口交"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.IS_FUTA_OR_MAN,
    }
)
def handle_target_mouth_and_chest_sex():
    """ 处理让对方乳交口交指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.TARGET_MOUTH_AND_CHEST_SEX
    character_data.state = constant.CharacterStatus.STATUS_TARGET_MOUTH_AND_CHEST_SEX
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.TARGET_VACCUM_MOUTH_SEX,
    constant.InstructType.SEX,
    _("让对方真空口交"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.IS_FUTA_OR_MAN,
    }
)
def handle_target_vaccum_mouth_sex():
    """ 处理让对方真空口交指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.TARGET_VACCUM_MOUTH_SEX
    character_data.state = constant.CharacterStatus.STATUS_TARGET_VACCUM_MOUTH_SEX
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.TARGET_DEEP_MOUTH_SEX,
    constant.InstructType.SEX,
    _("让对方深喉"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.IS_FUTA_OR_MAN,
    }
)
def handle_target_deep_mouth_sex():
    """ 处理让对方深喉指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.TARGET_DEEP_MOUTH_SEX
    character_data.state = constant.CharacterStatus.STATUS_TARGET_DEEP_MOUTH_SEX
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.MOUTH_AND_HAND_SEX,
    constant.InstructType.SEX,
    _("给对方手交口交"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.TARGET_IS_FUTA_OR_MAN,
    }
)
def handle_mouth_and_hand_sex():
    """ 处理给对方手交口交指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.MOUTH_AND_HAND_SEX
    character_data.state = constant.CharacterStatus.STATUS_MOUTH_AND_HAND_SEX
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.MOUTH_AND_CHEST_SEX,
    constant.InstructType.SEX,
    _("给对方乳交口交"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.TARGET_IS_FUTA_OR_MAN,
    }
)
def handle_mouth_and_chest_sex():
    """ 处理给对方乳交口交指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.MOUTH_AND_CHEST_SEX
    character_data.state = constant.CharacterStatus.STATUS_MOUTH_AND_CHEST_SEX
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.VACCUM_MOUTH_SEX,
    constant.InstructType.SEX,
    _("给对方真空口交"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.TARGET_IS_FUTA_OR_MAN,
    }
)
def handle_vaccum_mouth_sex():
    """ 处理让对方真空口交指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.VACCUM_MOUTH_SEX
    character_data.state = constant.CharacterStatus.STATUS_VACCUM_MOUTH_SEX
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.DEEP_MOUTH_SEX,
    constant.InstructType.SEX,
    _("给对方深喉"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.TARGET_IS_FUTA_OR_MAN,
    }
)
def handle_deep_mouth_sex():
    """ 处理给对方深喉指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.DEEP_MOUTH_SEX
    character_data.state = constant.CharacterStatus.STATUS_DEEP_MOUTH_SEX
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.SIX_NINE_SEX,
    constant.InstructType.SEX,
    _("69式"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
    }
)
def handle_six_nine_sex():
    """ 处理给对方深喉指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.SIX_NINE_SEX
    character_data.state = constant.CharacterStatus.STATUS_SIX_NINE_SEX
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.INSERTION_VAGINAL,
    constant.InstructType.SEX,
    _("插入阴道"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.IS_FUTA_OR_MAN,
        constant.Premise.TARGET_IS_FUTA_OR_WOMAN,
    }
)
def handle_insertion_vaginal():
    """ 处理插入阴道指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.INSERTION_VAGINAL
    character_data.state = constant.CharacterStatus.STATUS_INSERTION_VAGINAL
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.HITTING_UTERUS,
    constant.InstructType.SEX,
    _("撞击子宫"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.IS_FUTA_OR_MAN,
        constant.Premise.TARGET_IS_FUTA_OR_WOMAN,
    }
)
def handle_hitting_uterus():
    """ 处理撞击子宫指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.HITTING_UTERUS
    character_data.state = constant.CharacterStatus.STATUS_HITTING_UTERUS
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.RIDING_INSERTION_VAGINAL,
    constant.InstructType.SEX,
    _("骑乘位插入阴道"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.IS_FUTA_OR_MAN,
        constant.Premise.TARGET_IS_FUTA_OR_WOMAN,
    }
)
def handle_riding_insertion_vaginal():
    """ 处理骑乘位插入阴道指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.RIDING_INSERTION_VAGINAL
    character_data.state = constant.CharacterStatus.STATUS_RIDING_INSERTION_VAGINAL
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.BACK_INSERTION_VAGINAL,
    constant.InstructType.SEX,
    _("背后位插入阴道"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.IS_FUTA_OR_MAN,
        constant.Premise.TARGET_IS_FUTA_OR_WOMAN,
    }
)
def handle_back_insertion_vaginal():
    """ 处理背后位插入阴道指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.RIDING_INSERTION_VAGINAL
    character_data.state = constant.CharacterStatus.STATUS_RIDING_INSERTION_VAGINAL
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.BACK_RIDING_INSERTION_VAGINAL,
    constant.InstructType.SEX,
    _("背后位骑乘插入阴道"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.IS_FUTA_OR_MAN,
        constant.Premise.TARGET_IS_FUTA_OR_WOMAN,
    }
)
def handle_back_riding_insertion_vaginal():
    """ 处理背后位骑乘插入阴道指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.BACK_RIDING_INSERTION_VAGINAL
    character_data.state = constant.CharacterStatus.STATUS_BACK_RIDING_INSERTION_VAGINAL
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.NO_PENIS_SEX,
    constant.InstructType.SEX,
    _("磨豆腐"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.IS_WOMAN,
        constant.Premise.TARGET_IS_WOMAN,
    }
)
def handle_no_penix_sex():
    """ 处理磨豆腐指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.NO_PENIS_SEX
    character_data.state = constant.CharacterStatus.STATUS_NO_PENIS_SEX
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.INSERTION_ANUS,
    constant.InstructType.SEX,
    _("肛交"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.IS_FUTA_OR_MAN,
    }
)
def handle_insertion_anus():
    """ 处理肛交指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.INSERTION_ANUS
    character_data.state = constant.CharacterStatus.STATUS_INSERTION_ANUS
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.BACK_INSERTION_ANUS,
    constant.InstructType.SEX,
    _("背后位肛交"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.IS_FUTA_OR_MAN,
    }
)
def handle_back_insertion_anus():
    """ 处理背后位肛交指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.BACK_INSERTION_ANUS
    character_data.state = constant.CharacterStatus.STATUS_BACK_INSERTION_ANUS
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.RIDING_INSERTION_ANUS,
    constant.InstructType.SEX,
    _("骑乘位肛交"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.IS_FUTA_OR_MAN,
    }
)
def handle_riding_insertion_anus():
    """ 处理骑乘位肛交指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.RIDING_INSERTION_ANUS
    character_data.state = constant.CharacterStatus.STATUS_RIDING_INSERTION_ANUS
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.BACK_RIDING_INSERTION_ANUS,
    constant.InstructType.SEX,
    _("背后位骑乘肛交"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
        constant.Premise.IS_FUTA_OR_MAN,
    }
)
def handle_back_riding_insertion_anus():
    """ 处理背后位骑乘肛交指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.BACK_RIDING_INSERTION_ANUS
    character_data.state = constant.CharacterStatus.STATUS_BACK_RIDING_INSERTION_ANUS
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.SEX_END,
    constant.InstructType.SEX,
    _("结束做爱"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_PASSIVE_SEX,
    }
)
def handle_sex_end():
    """ 处理结束做爱指令 """
    character_data: game_type.Character = cache.character_data[0]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.passive_sex = 0
