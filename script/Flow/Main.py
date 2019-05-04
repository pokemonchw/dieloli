from script.Core import CacheContorl,GameInit,PyCmd
from script.Design import AttrHandle
from script.Panel import MainFramePanel
from script.Flow import SeeCharacterAttr,SeeCharacterList,Shop,ChangeClothes,GameSetting,SaveHandleFrame,InScene,GameHelp

# 游戏主页
def mainFrame_func():
    inputS = []
    flowReturn = MainFramePanel.mainFramePanel()
    inputS = inputS + flowReturn
    askForMainFrame(inputS)

# 主页控制流程
def askForMainFrame(ansList):
    characterId = CacheContorl.characterData['characterId']
    characterData = AttrHandle.getAttrData(characterId)
    characterName = characterData['Name']
    ans = GameInit.askfor_All(ansList)
    PyCmd.clr_cmd()
    if ans == characterName:
        mainFrameSeeAttrPanel()
    elif ans == '0':
        InScene.getInScene_func()
    elif ans == '1':
        SeeCharacterList.seeCharacterList_func('MainFramePanel')
    elif ans == '2':
        ChangeClothes.changeCharacterClothes(characterId)
    elif ans == '3':
        Shop.shopMainFrame_func()
    elif ans == '4':
        GameSetting.changeGameSetting_func()
    elif ans == '5':
        GameHelp.gameHelp_func()
    elif ans == '6':
        SaveHandleFrame.establishSave_func('MainFramePanel')
    elif ans == '7':
        SaveHandleFrame.loadSave_func('MainFramePanel')

# 游戏主页查看属性流程
def mainFrameSeeAttrPanel():
    SeeCharacterAttr.seeAttrOnEveryTime_func('MainFramePanel')
