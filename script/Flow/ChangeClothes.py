from script.Panel import ChangeClothesPanel
from script.Core import FlowHandle

# 更换角色服装流程
def changePlayerClothes(playerId):
    ChangeClothesPanel.seePlayerWearClothes(playerId)
    inputS = ChangeClothesPanel.seePlayerWearClothesCmd()
    yrn = FlowHandle.askfor_All(inputS)
    if yrn == '0':
        from script.Flow import Main
        Main.mainFrame_func()
