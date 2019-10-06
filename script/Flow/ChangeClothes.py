from script.Panel import ChangeClothesPanel
from script.Core import FlowHandle,CacheContorl,GameConfig,PyCmd
from script.Design import Clothing

def changeCharacterClothes():
    '''
    更换角色服装流程
    '''
    characterId = CacheContorl.characterData['characterId']
    characterClothingData = CacheContorl.characterData['character'][characterId]['Clothing']
    ChangeClothesPanel.seeCharacterWearClothesInfo(characterId)
    cmdList1 = ChangeClothesPanel.seeCharacterWearClothes(characterId,True)
    startId = len(characterClothingData.keys())
    inputS = ChangeClothesPanel.seeCharacterWearClothesCmd(startId)
    inputS = cmdList1 + inputS
    yrn = FlowHandle.askfor_All(inputS)
    PyCmd.clr_cmd()
    if yrn == str(startId):
        CacheContorl.nowFlowId = 'main'
    else:
        clothingType = list(Clothing.clothingTypeTextList.keys())[int(yrn)]
        seeCharacterClothesList(clothingType)

def seeCharacterClothesList(clothingType:str):
    '''
    查看角色服装列表流程
    Keyword arguments:
    clothingType -- 服装类型
    '''
    clothingTypeList = list(Clothing.clothingTypeTextList.keys())
    while(True):
        nowClothingTypeIndex = clothingTypeList.index(clothingType)
        upTypeId = nowClothingTypeIndex - 1
        if nowClothingTypeIndex == 0:
            upTypeId = len(clothingTypeList) - 1
        nextTypeId = nowClothingTypeIndex + 1
        if nowClothingTypeIndex == len(clothingTypeList) - 1:
            nextTypeId = 0
        upType = clothingTypeList[upTypeId]
        nextType = clothingTypeList[nextTypeId]
        characterId = CacheContorl.characterData['characterId']
        ChangeClothesPanel.seeCharacterClothesInfo(characterId)
        pageMax = getCharacterClothesPageMax(characterId,clothingType)
        inputS = ChangeClothesPanel.seeCharacterClothesPanel(characterId,clothingType,pageMax)
        startId = len(inputS)
        inputS += ChangeClothesPanel.seeCharacterClothesCmd(startId,clothingType)
        yrn = FlowHandle.askfor_All(inputS)
        yrn = int(yrn)
        PyCmd.clr_cmd()
        nowPageId = int(CacheContorl.panelState["SeeCharacterClothesPanel"])
        if yrn == startId:
            clothingType = upType
        elif yrn == startId + 1:
            if nowPageId == 0:
                CacheContorl.panelState['SeeCharacterClothesPanel'] = str(pageMax)
            else:
                CacheContorl.panelState['SeeCharacterClothesPanel'] = str(nowPageId - 1)
        elif yrn == startId + 2:
            break
        elif yrn == startId + 3:
            if nowPageId == pageMax:
                CacheContorl.panelState['SeeCharacterClothesPanel'] = '0'
            else:
                CacheContorl.panelState['SeeCharacterClothesPanel'] = str(nowPageId + 1)
        elif yrn == startId + 4:
            clothingType = nextType
        else:
            clothingId = list(CacheContorl.characterData['character'][characterId]['Clothing'][clothingType].keys())[yrn]
            askSeeClothingInfo(clothingType,clothingId)

def askSeeClothingInfo(clothingType:str,clothingId:str):
    '''
    确认查看服装详细信息流程
    Keyword arguments:
    clothingType -- 服装类型
    clothingId -- 服装id
    '''
    yrn = int(ChangeClothesPanel.askSeeClothingInfoPanel())

def getCharacterClothesPageMax(characterId:str,clothingType:str):
    '''
    计算角色某类型服装列表页数
    Keyword arguments:
    characterId -- 角色Id
    clothingType -- 服装类型
    '''
    clothingMax = len(CacheContorl.characterData['character'][characterId]['Clothing'][clothingType].keys())
    pageIndex = GameConfig.see_character_clothes_max
    if clothingMax - pageIndex < 0:
        return 0
    elif clothingMax % pageIndex == 0:
        return clothingMax / pageIndex - 1
    else:
        return int(clothingMax / pageIndex)
