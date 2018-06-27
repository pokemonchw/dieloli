from script.Core import TextLoading,EraPrint,CacheContorl
from script.Design import AttrHandle,ProportionalBar

# 用于输出角色血条和蓝条的方法（占一行，自动居中）
def printHpAndMpBar(playerId):
    playerData = AttrHandle.getAttrData(playerId)
    playerHitPoint = playerData['HitPoint']
    playerMaxHitPoint = playerData['HitPointMax']
    hitPointText = TextLoading.getTextData(TextLoading.stageWordId, '8')
    hitPointBar = ProportionalBar.getProportionalBar(hitPointText, playerMaxHitPoint, playerHitPoint, 'hpbar')
    playerManaPoint = playerData['ManaPoint']
    playerMaxManaPoint = playerData['ManaPointMax']
    manaPointText = TextLoading.getTextData(TextLoading.stageWordId, '9')
    manaPointBar = ProportionalBar.getProportionalBar(manaPointText, playerMaxManaPoint, playerManaPoint, 'mpbar')
    hpmpBarList = [hitPointBar, manaPointBar]
    EraPrint.p('\n')
    EraPrint.plist(hpmpBarList, 2, 'center')
    EraPrint.p('\n')

# 用于获取角色血条蓝条（无比例图）的方法
def getHpAndMpText(playerId):
    playerId = str(playerId)
    playerData = CacheContorl.playObject['object'][playerId]
    playerHitPoint = playerData['HitPoint']
    playerMaxHitPoint = playerData['HitPointMax']
    hitPointText = TextLoading.getTextData(TextLoading.stageWordId, '8')
    hpText = hitPointText + '(' + str(playerHitPoint) + '/' + str(playerMaxHitPoint) + ')'
    playerManaPoint = playerData['ManaPoint']
    playerMaxManaPoint = playerData['ManaPointMax']
    manaPointText = TextLoading.getTextData(TextLoading.stageWordId, '9')
    mpText = manaPointText + '(' + str(playerManaPoint) + '/' + str(playerMaxManaPoint) + ')'
    return hpText + ' ' + mpText