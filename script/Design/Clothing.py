from script.Core import CacheContorl,TextLoading,ValueHandle
import random
import math

def creatorSuit(suitName:str,sex:str) -> dict:
    '''
    创建套装
    Keyword arguments:
    suitName -- 套装模板
    sex -- 性别模板
    '''
    suitData = TextLoading.getTextData(TextLoading.equipmentPath,'Suit')[suitName][sex]
    newSuitData = {clothing:creatorClothing(suitData[clothing]) for clothing in suitData if suitData[clothing] != ''}
    return newSuitData

def creatorClothing(clothingName:str) -> dict:
    '''
    创建服装的基础函数
    Keyword arguments:
    clothingName -- 服装名字
    '''
    clothingData = {}
    clothingData['Sexy'] = random.randint(1,1000)
    clothingData['Handsome'] = random.randint(1,1000)
    clothingData['Elegant'] = random.randint(1,1000)
    clothingData['Fresh'] = random.randint(1,1000)
    clothingData['Sweet'] = random.randint(1,1000)
    clothingData['Warm'] = random.randint(0,30)
    clothingData['Price'] = sum([clothingData[x] for x in clothingData])
    setClothintEvaluationText(clothingData)
    clothingData['Cleanliness'] = 100
    clothingData.update(CacheContorl.clothingTypeData[clothingName])
    return clothingData

def chracterPutOnClothing(characterId:str):
    '''
    角色自动选择并穿戴服装
    Keyword arguments:
    characterId -- 角色id
    '''
    putOnData = {}
    collocationData = {}
    clothingNameData = {}
    characterClothingData = CacheContorl.characterData['character'][characterId]['creatorClothing']
    for nowClothingType in characterClothingData:
        for clothing in nowClothingData[nowClothingType]:
            if nowClothingData[nowCollocationalType][clothing]['Name'] not in clothingNameData:
                clothingNameData.setdefault(nowClothingData[nowCollocationalType][clothing]['Name'],{clothing:characterClothingData[nowClothingType][clothing]['Price'] + characterClothingData[nowClothingType][clothing]['Cleanliness']})
            else:
                clothingNameData[nowClothingData[nowCollocationalType][clothing]['Name']][clothing] = characterClothingData[nowClothingType][clothing]['Price'] + characterClothingData[nowClothingType][clothing]['Cleanliness']
        characterClothingData[nowClothingType] = {n:ValueHandle.sortedDictForValues(characterClothingData[nowClothingType][n]) for n in characterClothingData[nowClothingType]}
    clothingPriceData = {t:{c:characterClothingData[t][c]['Price'] + characterClothingData[t][c]['Cleanliness'] for c in characterClothingData[t]} for t in characterClothingData}
    for clothingType in clothingNameData:
        for clothingName in clothingNameData[clothingType]:
            nowClothingId = list(clothingNameData[clothingType][clothingName].keys())[-1]
            nowClothingData = characterClothingData[clothingType][clothingId]
            clothingCollocationTypeData = clothingData['CollocationalRestriction']
            collocationData[nowClothingId] = {}
            collocationData[nowClothingId]['Price']  = 0
            for nowCollocationalType in clothingCollocationTypeData:
                nowCollocationalRestrict = clothingCollocationTypeData[nowCollocationalType]
                if nowCollocationalRestrict == 'Precedence':
                    collocationData[nowClothingId][nowCollocationalType] = ''
                    precedenceList = {}
                    for precedenceClothing in characterClothingData[nowClothingType][nowClothingId]['Collocation'][nowCollocationalType]:
                        if precedenceClothing in characterClothingData and characterData[precedenceClothing] != {}:
                            precedenceList[precedenceClothing] = list(clothingNameData[nowCollocationalType][characterClothingData[nowClothingType][nowClothingId]['Collocation'][nowCollocationalType]].keys())[-1]
                    if precedenceList == {}:
                        for tooTooClothing in clothingPriceData[nowCollocationalType]:
                            if characterClothingData[nowCollocationalType][tooTooClothing]['CollocationalRestrict'][clothingType] == 'Usually':
                                collocationData[nowClothingId][nowCollocationalType] = tooTooClothing
                                collocationData[nowClothingId]['Price'] += clothingPriceData[tooTooClothing]
                                break
                            elif characterClothingData[nowCollocationalType][tooTooClothing]['CollocationalRestrict'][clothingType] == 'Must':
                                if clothingName in characterClothingData[nowCollocationalType][tooTooClothing]['Collocation'][nowClothingType]:
                                    collocationData[nowClothingId][nowCollocationalType] = tooTooClothing
                                    collocationData[nowClothingId]['Price'] += clothingPriceData[tooTooClothing]
                                    break
                        if collocationData[nowClothingId][nowCollocationalType] == '':
                            collocationData[nowClothingId] = 'None'
                            break
                    else:
                        precedenceList = ValueHandle.sortedDictForValues(precedenceList)
                        collocationData[nowClothingId][nowCollocationalType] = list(precedenceList.keys())[-1]
                elif nowCollocationalRestrict == 'Usually':
                    for tooTooClothing in clothingPriceData[nowCollocationalType]:
                        if characterClothingData[nowCollocationalType][tooTooClothing]['CollocationalRestrict'][clothingType] == 'Usually':
                            collocationData[nowClothingId][nowCollocationalType] = tooTooClothing
                            collocationData[nowClothingId]['Price'] += clothingPriceData[tooTooClothing]
                            break
                        elif characterClothingData[nowCollocationalType][tooTooClothing]['CollocationalRestrict'][clothingType] == 'Must':
                            if clothingName in characterClothingData[nowCollocationalType][tooTooClothing]['Collocation'][nowClothingType]:
                                collocationData[nowClothingId][nowCollocationalType] = tooTooClothing
                                collocationData[nowClothingId]['Price'] += clothingPriceData[tooTooClothing]
                                break
                    if collocationData[nowClothingId][nowCollocationalType] == '':
                        collocationData[nowClothingId] = 'None'
                        break
                elif nowCollocationalRestrict == 'None':
                    collocationData[nowClothingId][nowCollocationalType] = 'None'
                elif nowCollocationalRestrict == 'Ornone':
                    precedenceList = {}
                    for precedenceClothing in characterClothingData[nowClothingType][nowClothingId]['Collocation'][nowCollocationalType]:
                        if precedenceClothing in characterClothingData and characterData[precedenceClothing] != {}:
                            precedenceList[precedenceClothing] = list(clothingNameData[nowCollocationalType][characterClothingData[nowClothingType][nowClothingId]['Collocation'][nowCollocationalType]].keys())[-1]
                    if precedenceList == {}:
                        collocationData[nowClothingId][nowCollocationalType] = 'None'
                    else:
                        precedenceList = ValueHandle.sortedDictForValues(precedenceList)
                        collocationData[nowClothingId][nowCollocationalType] = list(precedenceList.keys())[-1]
                        collocationData[nowClothingId]['Price'] += clothingPriceData[list(precedenceList.keys())[-1]]
                elif nowCollocationalRestrict == 'Must':
                    precedenceList = {}
                    for precedenceClothing in characterClothingData[nowClothingType][nowClothingId]['Collocation'][nowCollocationalType]:
                        if precedenceClothing in characterClothingData and characterData[precedenceClothing] != {}:
                            precedenceList[precedenceClothing] = list(clothingNameData[nowCollocationalType][characterClothingData[nowClothingType][nowClothingId]['Collocation'][nowCollocationalType]].keys())[-1]
                    if precedenceList == {}:
                        collocationData[nowClothingId] = 'None'
                    else:
                        precedenceList = ValueHandle.sortedDictForValues(precedenceList)
                        collocationData[nowClothingId][nowCollocationalType] = list(precedenceList.keys())[-1]
                        collocationData[nowClothingId]['Price'] += clothingPriceData[list(precedenceList.keys())[-1]]

clothingEvaluationTextList = [TextLoading.getTextData(TextLoading.stageWordPath,str(k)) for k in range(102,112)]
clothingTagList = [TextLoading.getTextData(TextLoading.stageWordPath,str(k)) for k in range(112,117)]
def setClothintEvaluationText(clothingData:dict):
    '''
    设置服装的评价文本
    Keyword arguments:
    clothingData -- 服装数据
    '''
    clothingAttrData = [clothingData['Sexy'],clothingData['Handsome'],clothingData['Elegant'],clothingData['Fresh'],clothingData['Sweet']]
    clothingAttrMax = sum(clothingAttrData)
    clothingEvaluationText = clothingEvaluationTextList[math.floor(clothingData['Price'] / 480)]
    clothingTagText = clothingTagList[clothingAttrData.index(max(clothingAttrData)) - 1]
    clothingData['Evaluation'] = clothingEvaluationText
    clothingData['Tag'] = clothingTagText
