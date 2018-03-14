import core.TextLoading as textload
import script.ProportionalBar as proportionalbar
import core.EraPrint as eprint
import script.AttrHandle as attrhandle

# 用于输出角色血条和蓝条的方法（占一行，自动居中）
def printHpAndMpBar(playerId):
    playerData = attrhandle.getAttrData(playerId)
    playerHitPoint = playerData['HitPoint']
    playerMaxHitPoint = playerData['HitPointMax']
    hitPointText = textload.getTextData(textload.stageWordId, '8')
    hitPointBar = proportionalbar.getProportionalBar(hitPointText, playerMaxHitPoint, playerHitPoint, 'hpbar')
    playerManaPoint = playerData['ManaPoint']
    playerMaxManaPoint = playerData['ManaPointMax']
    manaPointText = textload.getTextData(textload.stageWordId, '9')
    manaPointBar = proportionalbar.getProportionalBar(manaPointText, playerMaxManaPoint, playerManaPoint, 'mpbar')
    hpmpBarList = [hitPointBar, manaPointBar]
    eprint.p('\n')
    eprint.plist(hpmpBarList, 2, 'center')
    eprint.p('\n')

