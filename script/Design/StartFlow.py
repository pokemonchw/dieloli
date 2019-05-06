# -*- coding: UTF-8 -*-
from script.Core import PyCmd,CacheContorl,FlowHandle
from script.Flow import CreatorCharacter,SaveHandleFrame,TitleFrame,Main,SeeCharacterAttr,InScene,SeeCharacterList,ChangeClothes,SeeMap,Shop,GameSetting,GameHelp

flowData = {
    "title_frame":TitleFrame.titleFrame_func,
    "creator_character":CreatorCharacter.inputName_func,
    "load_save":SaveHandleFrame.loadSave_func,
    "establish_save":SaveHandleFrame.establishSave_func,
    "main":Main.mainFrame_func,
    "see_character_attr":SeeCharacterAttr.seeAttrOnEveryTime_func,
    "in_scene":InScene.getInScene_func,
    "see_character_list":SeeCharacterList.seeCharacterList_func,
    "change_clothes":ChangeClothes.changeCharacterClothes,
    "see_map":SeeMap.seeMapFlow,
    "acknowledgment_attribute":SeeCharacterAttr.acknowledgmentAttribute_func,
    'shop':Shop.shopMainFrame_func,
    'game_setting':GameSetting.changeGameSetting_func,
    'game_help':GameHelp.gameHelp_func
}

# 游戏主流程
def startFrame():
    nowFlowId = ''
    FlowHandle.initCache()
    while(True):
        if nowFlowId != CacheContorl.nowFlowId:
            nowFlowId = CacheContorl.nowFlowId
            PyCmd.clr_cmd()
            flowData[nowFlowId]()
