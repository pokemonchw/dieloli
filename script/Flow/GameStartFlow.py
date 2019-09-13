from script.Design import GameTime,AttrCalculation,CharacterHandle,MapHandle,Course,Interest,Clothing
from script.Core import CacheContorl
from script.Panel import SeeCharacterAttrPanel
import uuid

def initGameStart():
    '''
    用于结束角色创建正式开始游戏的初始化流程
    '''
    GameTime.initTime()
    AttrCalculation.setAttrOver('0')
    Course.initPhaseCourseHour()
    CharacterHandle.initCharacterList()
    characterSuit = Clothing.creatorSuit('Uniform',CacheContorl.characterData['character']['0']['Sex'])
    for clothing in characterSuit:
        CacheContorl.characterData['character']['0']['Clothing'][clothing][uuid.uuid1()] = characterSuit[clothing]
    Interest.initCharacterInterest()
    Course.initCharacterKnowledge()
    Course.initClassTeacher()
    Course.initClassTimeTable()
    SeeCharacterAttrPanel.initShowAttrPanelList()
    characterPosition = CacheContorl.characterData['character']['0']['Position']
    MapHandle.characterMoveScene(['0'],characterPosition,'0')
    CacheContorl.nowFlowId = 'main'
