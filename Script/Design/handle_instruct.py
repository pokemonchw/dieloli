from Script.Core import text_loading, era_print, constant, cache_contorl
from Script.Design import game_time, update


def handle_unknown_instruct():
    era_print.line_feed_print(
        text_loading.get_text_data(constant.FilePath.MESSAGE_PATH, "42")
    )


def handle_rest():
    """
    处理休息指令
    """
    cache_contorl.character_data[0].hit_point += 50
    cache_contorl.character_data[0].mana_point += 100
    if (
        cache_contorl.character_data[0].hit_point
        > cache_contorl.character_data[0].hit_point_max
    ):
        cache_contorl.character_data[
            0
        ].hit_point = cache_contorl.character_data[0].hit_point_max
    if (
        cache_contorl.character_data[0].mana_point
        > cache_contorl.character_data[0].mana_point_max
    ):
        cache_contorl.character_data[
            0
        ].mana_point = cache_contorl.character_data[0].mana_point_max
    game_time.sub_time_now(10)
    update.game_update_flow()


handle_instruct_data = {
    "Gossip": handle_unknown_instruct,
    "Praise": handle_unknown_instruct,
    "Vilify": handle_unknown_instruct,
    "Deception": handle_unknown_instruct,
    "StrokeHead": handle_unknown_instruct,
    "StrokeFace": handle_unknown_instruct,
    "PinchFace": handle_unknown_instruct,
    "PatShoulder": handle_unknown_instruct,
    "AroundShoulder": handle_unknown_instruct,
    "AroundWaist": handle_unknown_instruct,
    "Handshake": handle_unknown_instruct,
    "JoinHands": handle_unknown_instruct,
    "Embrace": handle_unknown_instruct,
    "Kiss": handle_unknown_instruct,
    "KneePillow": handle_unknown_instruct,
    "PillowShoulder": handle_unknown_instruct,
    "AskStrokeHead": handle_unknown_instruct,
    "AskStrokeFace": handle_unknown_instruct,
    "AskFace": handle_unknown_instruct,
    "AskEmbrace": handle_unknown_instruct,
    "Paint": handle_unknown_instruct,
    "PoetryRecital": handle_unknown_instruct,
    "Sing": handle_unknown_instruct,
    "PlayGuitar": handle_unknown_instruct,
    "PlayHarmonica": handle_unknown_instruct,
    "PlayBambooFlute": handle_unknown_instruct,
    "Dance": handle_unknown_instruct,
    "PlayBasketball": handle_unknown_instruct,
    "PlayFootball": handle_unknown_instruct,
    "PlayTableTennis": handle_unknown_instruct,
    "Swim": handle_unknown_instruct,
    "Rpg": handle_unknown_instruct,
    "UnarmedCombat": handle_unknown_instruct,
    "CloseCombat": handle_unknown_instruct,
    "Shooting": handle_unknown_instruct,
    "Dodge": handle_unknown_instruct,
    "Peeping": handle_unknown_instruct,
    "LoveStroke": handle_unknown_instruct,
    "StrokeBosom": handle_unknown_instruct,
    "StrokeClitoris": handle_unknown_instruct,
    "StrokePenis": handle_unknown_instruct,
    "StrokeAnus": handle_unknown_instruct,
    "StrokeThigh": handle_unknown_instruct,
    "LickingBosom": handle_unknown_instruct,
    "LickingFoot": handle_unknown_instruct,
    "Masturbation": handle_unknown_instruct,
    "ReadBook": handle_unknown_instruct,
    "SelfStudy": handle_unknown_instruct,
    "AttendLectures": handle_unknown_instruct,
    "Teaching": handle_unknown_instruct,
    "CombatTraining": handle_unknown_instruct,
    "ShootingTraining": handle_unknown_instruct,
    "DodgeTraining": handle_unknown_instruct,
    "Rest": handle_rest,
    "Doze": handle_unknown_instruct,
    "Siesta": handle_unknown_instruct,
    "Sleep": handle_unknown_instruct,
}
