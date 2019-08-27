from script.Core import TextLoading,EraPrint,CacheContorl
from script.Design import AttrHandle,ProportionalBar

def printHpAndMpBar(characterId):
    '''
    绘制角色的hp和mp(有比例图)，自动居中处理，结尾换行
    Keyword arguments:
    characterId -- 角色id
    '''
    characterData = AttrHandle.getAttrData(characterId)
    characterHitPoint = characterData['HitPoint']
    characterMaxHitPoint = characterData['HitPointMax']
    hitPointText = TextLoading.getTextData(TextLoading.stageWordPath, '8')
    hitPointBar = ProportionalBar.getProportionalBar(hitPointText, characterMaxHitPoint, characterHitPoint, 'hpbar')
    characterManaPoint = characterData['ManaPoint']
    characterMaxManaPoint = characterData['ManaPointMax']
    manaPointText = TextLoading.getTextData(TextLoading.stageWordPath, '9')
    manaPointBar = ProportionalBar.getProportionalBar(manaPointText, characterMaxManaPoint, characterManaPoint, 'mpbar')
    hpmpBarList = [hitPointBar, manaPointBar]
    EraPrint.p('\n')
    EraPrint.plist(hpmpBarList, 2, 'center')
    EraPrint.p('\n')

def getHpAndMpText(characterId):
    '''
    获取角色的hp和mp文本
    Keyword arguments:
    characterId -- 角色id
    '''
    characterId = str(characterId)
    characterData = CacheContorl.characterData['character'][characterId]
    characterHitPoint = characterData['HitPoint']
    characterMaxHitPoint = characterData['HitPointMax']
    hitPointText = TextLoading.getTextData(TextLoading.stageWordPath, '8')
    hpText = hitPointText + '(' + str(characterHitPoint) + '/' + str(characterMaxHitPoint) + ')'
    characterManaPoint = characterData['ManaPoint']
    characterMaxManaPoint = characterData['ManaPointMax']
    manaPointText = TextLoading.getTextData(TextLoading.stageWordPath, '9')
    mpText = manaPointText + '(' + str(characterManaPoint) + '/' + str(characterMaxManaPoint) + ')'
    return hpText + ' ' + mpText
