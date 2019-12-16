from script.Core import CacheContorl,GameInit,PyCmd
from script.Design import AttrHandle,Clothing
from script.Panel import MainFramePanel

mainFrameGotoData = {
    "0":'in_scene',
    "1":'see_character_list',
    "2":'change_clothes',
    "3":'wear_item',
    "4":'open_bag',
    "5":'shop',
    "6":'game_setting',
    "7":'game_help',
    "8":'establish_save',
    "9":'load_save',
}

def mainFrame_func():
    '''
    游戏主页控制流程
    '''
    inputS = []
    flowReturn = MainFramePanel.mainFramePanel()
    inputS = inputS + flowReturn
    characterId = CacheContorl.characterData['characterId']
    characterData = AttrHandle.getAttrData(characterId)
    characterName = characterData['Name']
    ans = GameInit.askfor_All(inputS)
    PyCmd.clr_cmd()
    CacheContorl.oldFlowId = 'main'
    if ans == characterName:
        CacheContorl.nowFlowId = 'see_character_attr'
    else:
        if ans == '0':
            Clothing.initCharcterClothintPutOn()
        CacheContorl.nowFlowId = mainFrameGotoData[ans]
