from script.Core import TextLoading,EraPrint,CacheContorl
from script.Design import AttrHandle,ProportionalBar

# 用于输出角色血条和蓝条的方法（占一行，自动居中）
def printHpAndMpBar(characterId):
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

# 用于获取角色血条蓝条（无比例图）的方法
def getHpAndMpText(characterId):
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
