from script.Panel import ChangeClothesPanel
from script.Core import FlowHandle,CacheContorl,GameConfig
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
    characterId = CacheContorl.characterData['characterId']
    ChangeClothesPanel.seeCharacterClothesInfo(characterId)
    pageMax = getCharacterClothesPageMax(characterId,clothingType)
    ChangeClothesPanel.seeCharacterClothesPanel(characterId,clothingType,pageMax)
    inputS = []

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
