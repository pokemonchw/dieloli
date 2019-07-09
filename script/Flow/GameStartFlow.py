from script.Design import GameTime,AttrCalculation,CharacterHandle,MapHandle,Course,Interest
from script.Core import CacheContorl
from script.Panel import SeeCharacterAttrPanel
import datetime

# 用于结束角色创建正式开始游戏的初始化流程
def initGameStart():
    GameTime.initTime()
    AttrCalculation.setAttrOver('0')
    Course.initPhaseCourseHour()
    # 0.48s
    time1 = datetime.datetime.now()
    CharacterHandle.initCharacterList()
    time2 = datetime.datetime.now()
    # 0.32s
    # 0.16s
    Interest.initCharacterInterest()
    # 0.21s
    # 0.13s
    Course.initCharacterKnowledge()
    #print(time2-time1)
    SeeCharacterAttrPanel.initShowAttrPanelList()
    characterPosition = CacheContorl.characterData['character']['0']['Position']
    MapHandle.characterMoveScene(['0'],characterPosition,'0')
    CacheContorl.nowFlowId = 'main'
