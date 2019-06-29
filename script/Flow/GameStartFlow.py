from script.Design import GameTime,AttrCalculation,CharacterHandle,MapHandle,Course
from script.Core import CacheContorl
from script.Panel import SeeCharacterAttrPanel

# 用于结束角色创建正式开始游戏的初始化流程
def initGameStart():
    GameTime.initTime()
    AttrCalculation.setAttrOver('0')
    CharacterHandle.initCharacterList()
    SeeCharacterAttrPanel.initShowAttrPanelList()
    Course.initCourse()
    characterPosition = CacheContorl.characterData['character']['0']['Position']
    MapHandle.characterMoveScene(['0'],characterPosition,'0')
    CacheContorl.nowFlowId = 'main'
