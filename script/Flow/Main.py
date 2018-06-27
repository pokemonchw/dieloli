from script.Core import CacheContorl,GameInit,PyCmd
from script.Design import AttrHandle
from script.Panel import MainFramePanel
from script.Flow import SeePlayerAttr,SeePlayerList,Shop,ChangeClothes,GameSetting,SaveHandleFrame,InScene

# 游戏主页
def mainFrame_func():
    inputS = []
    flowReturn = MainFramePanel.mainFramePanel()
    inputS = inputS + flowReturn
    askForMainFrame(inputS)

# 主页控制流程
def askForMainFrame(ansList):
    playerId = CacheContorl.playObject['objectId']
    playerData = AttrHandle.getAttrData(playerId)
    playerName = playerData['Name']
    ans = GameInit.askfor_All(ansList)
    PyCmd.clr_cmd()
    if ans == playerName:
        mainFrameSeeAttrPanel()
    elif ans == '0':
        InScene.getInScene_func()
    elif ans == '1':
        SeePlayerList.seePlayerList_func('MainFramePanel')
    elif ans == '2':
        ChangeClothes.changePlayerClothes(playerId)
    elif ans == '3':
        Shop.shopMainFrame_func()
    elif ans == '4':
        GameSetting.changeGameSetting_func()
    elif ans == '5':
        pass
    elif ans == '6':
        SaveHandleFrame.establishSave_func('MainFramePanel')
    elif ans == '7':
        SaveHandleFrame.loadSave_func('MainFramePanel')
        pass

# 游戏主页查看属性流程
def mainFrameSeeAttrPanel():
    SeePlayerAttr.seeAttrOnEveryTime_func('MainFramePanel')
    pass