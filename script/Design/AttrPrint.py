from script.Core import TextLoading,EraPrint,CacheContorl,GameConfig
from script.Design import ProportionalBar

pointTextData = {
    "HitPoint":TextLoading.getTextData(TextLoading.stageWordPath,'8'),
    "ManaPoint":TextLoading.getTextData(TextLoading.stageWordPath,'9')
}

def printHpAndMpBar(characterId:str):
    '''
    绘制角色的hp和mp(有比例图)，自动居中处理，结尾换行
    Keyword arguments:
    characterId -- 角色id
    '''
    hpBar = getHpOrMpBar(characterId,'HitPoint',GameConfig.text_width / 2 - 4)
    mpBar = getHpOrMpBar(characterId,'ManaPoint',GameConfig.text_width / 2 - 4)
    EraPrint.p('\n')
    EraPrint.plist([hpBar,mpBar], 2, 'center')
    EraPrint.p('\n')

def getHpOrMpBar(characterId:str,barId:str,textWidth:int):
    '''
    获取角色的hp或mp比例图，按给定宽度居中
    Keyword arguments:
    characterId -- 角色Id
    barId -- 绘制的比例条类型(hp/mp)
    textWidth -- 比例条总宽度
    '''
    characterData = CacheContorl.characterData['character'][characterId]
    if barId == 'HitPoint':
        characterPoint = characterData.HitPoint
        characterMaxPoint = characterData.HitPointMax
    else:
        characterPoint = characterData.ManaPoint
        characterMaxPoint = characterData.ManaPointMax
    pointText = pointTextData[barId]
    return ProportionalBar.getProportionalBar(pointText,characterMaxPoint,characterPoint,barId + 'bar',textWidth)

def getHpAndMpText(characterId:str) -> str:
    '''
    获取角色的hp和mp文本
    Keyword arguments:
    characterId -- 角色id
    '''
    characterId = str(characterId)
    characterData = CacheContorl.characterData['character'][characterId]
    characterHitPoint = characterData.HitPoint
    characterMaxHitPoint = characterData.HitPointMax
    hitPointText = TextLoading.getTextData(TextLoading.stageWordPath, '8')
    hpText = hitPointText + '(' + str(characterHitPoint) + '/' + str(characterMaxHitPoint) + ')'
    characterManaPoint = characterData.ManaPoint
    characterMaxManaPoint = characterData.ManaPointMax
    manaPointText = TextLoading.getTextData(TextLoading.stageWordPath, '9')
    mpText = manaPointText + '(' + str(characterManaPoint) + '/' + str(characterMaxManaPoint) + ')'
    return hpText + ' ' + mpText
