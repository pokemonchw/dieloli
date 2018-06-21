from Core import CacheContorl
from Design import ObjectBehavior

# npc行为总控制
def initObjectBehavior():
    npcList = ObjectBehavior.getCharacterIdList()
    for npc in npcList:
        objectStateBehavior(npc)

# npc状态判断
def objectStateBehavior(objectId):
    objectData = CacheContorl.playObject['object'][objectId]
