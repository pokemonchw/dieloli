from Core import CacheContorl
from Design import CharacterHandle

# npc行为总控制
def initObjectBehavior():
    npcList = CharacterHandle.getCharacterIdList()
    for npc in npcList:
        objectStateBehavior(npc)

# npc状态判断
def objectStateBehavior(objectId):
    objectData = CacheContorl.playObject['object'][objectId]
