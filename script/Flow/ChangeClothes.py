from script.Panel import ChangeClothesPanel
from script.Core import FlowHandle,CacheContorl

# 更换角色服装流程
def changeCharacterClothes():
    characterId = CacheContorl.characterData['characterId']
    ChangeClothesPanel.seeCharacterWearClothes(characterId)
    inputS = ChangeClothesPanel.seeCharacterWearClothesCmd()
    yrn = FlowHandle.askfor_All(inputS)
    if yrn == '0':
        CacheContorl.nowFlowId = 'main'
