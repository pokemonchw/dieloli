import core.CacheContorl as cache
import core.ValueHandle as valuehandle
import core.data as data
import os
import core.TextLoading as textload
import script.AttrCalculation as attr
from core.pycfg import gamepath
from core.GameConfig import language

# 初始化角色数据
def initCharacterList():
    characterListPath = os.path.join(gamepath,'data',language,'character')
    characterList = data.getPathList(characterListPath)
    for i in range(0,len(characterList)):
        attr.initTemporaryObject()
        playerId = str(i + 1)
        cache.playObject['object'][playerId] = cache.temporaryObject.copy()
        attr.setDefaultCache()
        characterDataName = characterList[i]
        characterAttrTemPath = os.path.join(characterListPath,characterDataName,'AttrTemplate.json')
        characterData = data._loadjson(characterAttrTemPath)
        characterName = characterData['Name']
        characterSex = characterData['Sex']
        characterSexTem = textload.getTextData(textload.temId,'TemList')[characterSex]
        cache.playObject['object'][playerId]['Sex'] = characterSex
        characterDataKeys = valuehandle.dictKeysToList(characterData)
        defaultAttr = attr.getAttr(characterSexTem)
        defaultAttr['Name'] = characterName
        defaultAttr['Sex'] = characterSex
        attr.setSexCache(characterSex)
        defaultAttr['Features'] = cache.featuresList.copy()
        if 'Age' in  characterDataKeys:
            ageTem = characterData['Age']
            characterAge = attr.getAge(ageTem)
            defaultAttr['Age'] = characterAge
        elif 'Features' in characterDataKeys:
            attr.setAddFeatures(characterData['Features'])
            defaultAttr['Features'] = cache.featuresList.copy()
        for keys in defaultAttr:
            cache.temporaryObject[keys] = defaultAttr[keys]
        cache.featuresList = {}
        cache.playObject['object'][playerId] = cache.temporaryObject.copy()
        cache.temporaryObject = cache.temporaryObjectBak.copy()
    pass

# 获取角色最大数量
def getCharacterIndexMax():
    playerData = cache.playObject['object']
    playerMax = valuehandle.indexDictKeysMax(playerData) - 1
    return playerMax

# 获取角色id列表
def getCharacterIdList():
    playerData = cache.playObject['object']
    playerList = valuehandle.dictKeysToList(playerData)
    return playerList