import script.Panel.MainFramePanel as mainframepanel
import core.CacheContorl as cache
import script.AttrHandle as attrhandle
import core.game as game
import core.PyCmd as pycmd
import script.flow.SeePlayerAttr as seeplayerattr
import script.flow.SeePlayerList as seeplayerlist
import script.flow.Shop as shop
import script.flow.ChangeClothes as changeclothes
import script.flow.GameSetting as gamesetting
import script.flow.SaveHandleFrame as savehandleframe
import script.flow.InScene as inscene

# 游戏主页
def mainFrame_func():
    inputS = []
    flowReturn = mainframepanel.mainFramePanel()
    inputS = inputS + flowReturn
    askForMainFrame(inputS)

# 主页控制流程
def askForMainFrame(ansList):
    playerId = cache.playObject['objectId']
    playerData = attrhandle.getAttrData(playerId)
    playerName = playerData['Name']
    ans = game.askfor_All(ansList)
    pycmd.clr_cmd()
    if ans == playerName:
        mainFrameSeeAttrPanel()
    elif ans == '0':
        inscene.getInScene_func()
    elif ans == '1':
        seeplayerlist.seePlayerList_func('MainFramePanel')
    elif ans == '2':
        changeclothes.changePlayerClothes(playerId)
    elif ans == '3':
        shop.shopMainFrame_func()
    elif ans == '4':
        gamesetting.changeGameSetting_func()
    elif ans == '5':
        pass
    elif ans == '6':
        savehandleframe.establishSave_func('MainFramePanel')
    elif ans == '7':
        savehandleframe.loadSave_func('MainFramePanel')
        pass

# 游戏主页查看属性流程
def mainFrameSeeAttrPanel():
    seeplayerattr.seeAttrOnEveryTime_func('MainFramePanel')
    pass