from script.Panel import ChangeClothesPanel
from script.Core import FlowHandle,CacheContorl

def changeCharacterClothes():
    '''
    更换角色服装流程
    '''
    characterId = CacheContorl.characterData['characterId']
    ChangeClothesPanel.seeCharacterWearClothes(characterId)
    inputS = ChangeClothesPanel.seeCharacterWearClothesCmd()
    yrn = FlowHandle.askfor_All(inputS)
    if yrn == '0':
        CacheContorl.nowFlowId = 'main'
