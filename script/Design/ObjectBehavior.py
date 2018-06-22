from Core import CacheContorl
from Design import CharacterHandle

# npc行为总控制
def initObjectBehavior():
    npcList = CharacterHandle.getCharacterIdList()
    for npc in npcList:
        objectStateJudge(npc)

# npc状态判断
def objectStateJudge(objectId):
    objectData = CacheContorl.playObject['object'][objectId]
    objectState = objectData['State']
    if objectState == 'Arder':
        objectArderBehavior(objectId,objectData)

# 休闲状态行为
def objectArderBehavior(objectId,objectData):
    objectAge = objectData['Age']
    objectAge = int(objectAge)
    if objectAge <= 18:
        studentArderBehavior(objectId,objectData)
    else:
        pass

# 学生休闲行为
def studentArderBehavior(objectId,objectData):
    pass