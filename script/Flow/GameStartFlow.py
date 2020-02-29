from script.Design import AttrCalculation,CharacterHandle,MapHandle,Course,Interest,Clothing,Nature
from script.Core import CacheContorl
from script.Panel import SeeCharacterAttrPanel
import uuid

def initGameStart():
    '''
    用于结束角色创建正式开始游戏的初始化流程
    '''
    CharacterHandle.initCharacterDormitory()
    CharacterHandle.initCharacterPosition()
    Course.initPhaseCourseHour()
    Interest.initCharacterInterest()
    Course.initCharacterKnowledge()
    Course.initClassTeacher()
    Course.initClassTimeTable()
    characterPosition = CacheContorl.characterData['character'][0].Position
    MapHandle.characterMoveScene(['0'],characterPosition,0)
    CacheContorl.nowFlowId = 'main'
