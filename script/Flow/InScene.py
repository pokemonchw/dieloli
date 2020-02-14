from script.Core import CacheContorl,GameInit,PyCmd,GameConfig
from script.Design import MapHandle,CharacterHandle,PanelStateHandle
from script.Panel import InScenePanel,SeeCharacterAttrPanel,InstructPanel
import math

def getInScene_func():
    '''
    用于进入场景界面的流程
    '''
    PyCmd.clr_cmd()
    scenePath = CacheContorl.characterData['character']['0']['Position']
    scenePathStr = MapHandle.getMapSystemPathStrForList(scenePath)
    MapHandle.sortSceneCharacterId(scenePathStr)
    CacheContorl.nowMap = MapHandle.getMapForPath(scenePath)
    sceneData = CacheContorl.sceneData[scenePathStr].copy()
    sceneCharacterList = sceneData['SceneCharacterData']
    if '0' not in sceneCharacterList:
        characterIdList = ['0']
        sceneCharacterList = sceneCharacterList + characterIdList
        CacheContorl.sceneData[scenePathStr]['SceneCharacterData'] = sceneCharacterList
    if len(sceneCharacterList) > 1 and CacheContorl.characterData['characterId'] == '0':
        nowNameList = MapHandle.getSceneCharacterNameList(scenePathStr)
        nowNameList.remove(CacheContorl.characterData['character']['0']['Name'])
        CacheContorl.characterData['characterId'] = MapHandle.getCharacterIdByCharacterName(nowNameList[0],scenePathStr)
        if CacheContorl.oldCharacterId != '0':
            CacheContorl.characterData['characterId'] = CacheContorl.oldCharacterId
            CacheContorl.oldCharacterId = '0'
    if len(sceneCharacterList) > 1:
        seeScene_func(True)
    else:
        seeScene_func(False)

def seeScene_func(judge:bool):
    '''
    用于查看当前场景界面的流程
    Keyword argument:
    judge -- 判断是否绘制角色列表界面的开关
    '''
    while(True):
        inputS = []
        InScenePanel.seeScenePanel()
        scenePath = CacheContorl.characterData['character']['0']['Position']
        scenePathStr = MapHandle.getMapSystemPathStrForList(scenePath)
        sceneCharacterNameList = MapHandle.getSceneCharacterNameList(scenePathStr)
        nameListMax = int(GameConfig.in_scene_see_player_max)
        changePageJudge = False
        if len(sceneCharacterNameList) == 1:
            CacheContorl.characterData['characterId'] = '0'
        inSceneCmdList1 = []
        if judge:
            if CacheContorl.panelState['SeeSceneCharacterListPage'] == '0':
                inputS = inputS + InScenePanel.seeSceneCharacterListPanel()
                if len(sceneCharacterNameList) > nameListMax:
                    inSceneCmdList1 = InScenePanel.changeSceneCharacterListPanel()
                    changePageJudge = True
            inputS.append('SeeSceneCharacterListPage')
        startId1 = len(inSceneCmdList1)
        InScenePanel.seeCharacterInfoPanel()
        SeeCharacterAttrPanel.seeCharacterHPAndMPInSence(CacheContorl.characterData['characterId'])
        SeeCharacterAttrPanel.seeCharacterStatusPanel(CacheContorl.characterData['characterId'])
        instructHead = InstructPanel.seeInstructHeadPanel()
        inSceneCmdList2 = InScenePanel.inSceneButtonPanel(startId1)
        if changePageJudge:
            inputS += inSceneCmdList1 + instructHead + inSceneCmdList2
        else:
            inputS += instructHead + inSceneCmdList2
        yrn = GameInit.askfor_All(inputS)
        PyCmd.clr_cmd()
        nowPage = int(CacheContorl.panelState['SeeSceneCharacterListPanel'])
        characterMax = CharacterHandle.getCharacterIndexMax() - 1
        pageMax = math.floor(characterMax / nameListMax)
        if yrn in sceneCharacterNameList:
            CacheContorl.characterData['characterId'] = MapHandle.getCharacterIdByCharacterName(yrn,scenePathStr)
        elif judge and yrn not in inSceneCmdList2 and yrn != 'SeeSceneCharacterListPage' and changePageJudge:
            if yrn == inSceneCmdList1[0]:
                CacheContorl.panelState['SeeSceneCharacterListPanel'] = 0
            elif yrn == inSceneCmdList1[1]:
                if int(nowPage) == 0:
                    CacheContorl.panelState['SeeSceneCharacterListPanel'] = pageMax
                else:
                    CacheContorl.panelState['SeeSceneCharacterListPanel'] = int(nowPage) - 1
            elif yrn == inSceneCmdList1[2]:
                CacheContorl.panelState['SeeSceneCharacterListPanel'] = InScenePanel.jumpCharacterListPagePanel()
            elif yrn == inSceneCmdList1[3]:
                if int(nowPage) == pageMax:
                    CacheContorl.panelState['SeeSceneCharacterListPanel'] = 0
                else:
                    CacheContorl.panelState['SeeSceneCharacterListPanel'] = int(nowPage) + 1
            elif yrn == inSceneCmdList1[4]:
                CacheContorl.panelState['SeeSceneCharacterListPanel'] = pageMax
        elif yrn == inSceneCmdList2[0]:
            CacheContorl.nowFlowId = 'see_map'
            nowMap = MapHandle.getMapForPath(CacheContorl.characterData['character']['0']['Position'])
            CacheContorl.nowMap = nowMap
            break
        elif yrn in [inSceneCmdList2[1],inSceneCmdList2[2]]:
            if yrn == inSceneCmdList2[2]:
                CacheContorl.oldCharacterId = CacheContorl.characterData['characterId']
                CacheContorl.characterData['characterId'] = '0'
            CacheContorl.nowFlowId = 'see_character_attr'
            CacheContorl.oldFlowId = 'in_scene'
            break
        elif yrn == 'SeeSceneCharacterListPage':
            PanelStateHandle.panelStateChange(yrn)
        elif yrn in instructHead:
            if CacheContorl.instructFilter[yrn] == 1:
                CacheContorl.instructFilter[yrn] = 0
            else:
                CacheContorl.instructFilter[yrn] = 1
