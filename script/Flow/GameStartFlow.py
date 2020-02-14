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
    characterSuit = Clothing.creatorSuit('Uniform',CacheContorl.characterData['character']['0']['Sex'])
    for clothing in characterSuit:
        CacheContorl.characterData['character']['0']['Clothing'][clothing][uuid.uuid1()] = characterSuit[clothing]
    Clothing.characterPutOnClothing('0')
    Interest.initCharacterInterest()
    Course.initCharacterKnowledge()
    Course.initClassTeacher()
    Course.initClassTimeTable()
    if 'Nature' not in CacheContorl.characterData['character']['0']:
        Nature.initCharacterNature('0')
    characterPosition = CacheContorl.characterData['character']['0']['Position']
    MapHandle.characterMoveScene(['0'],characterPosition,'0')
    CacheContorl.nowFlowId = 'main'
