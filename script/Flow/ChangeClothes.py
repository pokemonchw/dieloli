from script.Panel import ChangeClothesPanel
from script.Core import FlowHandle

# 更换角色服装流程
def changeCharacterClothes(characterId):
    ChangeClothesPanel.seeCharacterWearClothes(characterId)
    inputS = ChangeClothesPanel.seeCharacterWearClothesCmd()
    yrn = FlowHandle.askfor_All(inputS)
    if yrn == '0':
        from script.Flow import Main
        Main.mainFrame_func()
