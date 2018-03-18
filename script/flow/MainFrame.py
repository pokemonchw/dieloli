import script.Panel.MainFramePanel as mainframepanel
import core.CacheContorl as cache
import script.AttrHandle as attrhandle
import core.game as game
import core.PyCmd as pycmd
import script.flow.SeePlayerAttr as seeplayerattr
import script.flow.MapEvent as mapevent
import script.flow.SeeObjectListFlow as seeobjectlistflow
import script.flow.Shop as shop
import script.flow.ChangeClothes as changeclothes
import script.flow.GameSetting as gamesetting
import script.flow.SaveHandleFrame as savehandleframe

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
    if ans == playerName:
        pycmd.clr_cmd()
        mainFrameSeeAttrPanel()
    elif ans == '0':
        pycmd.clr_cmd()
        mapevent.playerOnScene_Func()
    elif ans == '1':
        pycmd.clr_cmd()
        seeobjectlistflow.seePlayerObjectList_func()
    elif ans == '2':
        pycmd.clr_cmd()
        changeclothes.changePlayerClothes(playerId)
    elif ans == '3':
        pycmd.clr_cmd()
        shop.shopMainFrame_func()
    elif ans == '4':
        pycmd.clr_cmd()
        gamesetting.changeGameSetting_func()
    elif ans == '5':
        pass
    elif ans == '6':
        savehandleframe.establishSave_func()
    elif ans == '7':
        pass

# 游戏主页查看属性流程
def mainFrameSeeAttrPanel():
    seeplayerattr.seeAttrInEveryTime_func()
    pass