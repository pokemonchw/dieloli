from script.Core import CacheContorl,TextLoading,EraPrint
from script.Design import AttrText,CmdButtonQueue

def seeCharacterNaturePanel(characterId:str):
    '''
    用于菜单中查看角色性格信息面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    seeCharacter(characterId,False)

def seeCharacterNatureChangePanel(characterId:str) -> list:
    '''
    用于查看和切换角色性格信息面板
    Keyword arguments:
    characterId -- 角色Id
    Return arguments:
    list -- 按钮列表
    '''
    return seeCharacter(characterId,True)

def seeCharacter(characterId:str,judge:bool) -> list:
    '''
    用于任何时候查看角色性格信息面板
    Keyword arguments:
    characterId -- 角色Id
    judge -- 绘制按钮校验
    Return arguments:
    list -- 按钮列表
    '''
    natureTextData = TextLoading.getGameData(TextLoading.naturePath)
    characterNature = CacheContorl.characterData['character'][characterId].Nature
    cmdList = []
    for nature in natureTextData:
        natureText = natureTextData[nature]['Name']
        if 'Good' in natureText:
            nowNatureValues = [characterNature[sonNature] for sonNature in natureTextData[nature]['Factor']]
            nowNatureValue = sum(nowNatureValues)
            nowNatureMax = len(nowNatureValues) * 100
            if nowNatureValue < nowNatureMax / 2:
                natureText = natureText['Bad']
            else:
                natureText = natureText['Good']
        EraPrint.sontitleprint(natureText)
        infoList = [natureTextData[nature]['Factor'][sonNature][judgeNatureGood(characterNature[sonNature])] for sonNature in natureTextData[nature]['Factor']]
        if judge:
            nowSonList = [son for son in natureTextData[nature]['Factor']]
            cmdList += nowSonList
            CmdButtonQueue.optionstr(None,len(nowSonList),'center',False,False,infoList,'',nowSonList)
        else:
            EraPrint.plist(infoList,len(infoList),'center')
    return cmdList

def judgeNatureGood(nature:int) -> str:
    '''
    校验性格倾向
    Keyword arguments:
    nature -- 性格数值
    Return arguments:
    str -- 好坏
    '''
    if nature < 50:
        return 'Bad'
    return 'Good'
