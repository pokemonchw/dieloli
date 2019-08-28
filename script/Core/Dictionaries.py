from script.Core import CacheContorl

def handleText(string:str) -> str:
    '''
    对文本中的宏进行转义处理
    Keyword arguments:
    string -- 需要进行转义处理的文本
    '''
    characterId = CacheContorl.characterData['characterId']
    if characterId != '':
        characterName = CacheContorl.characterData['character'][characterId]['Name']
        characterNickName = CacheContorl.characterData['character'][characterId]['NickName']
        characterSelfName = CacheContorl.characterData['character'][characterId]['SelfName']
        return string.format(
            Name=characterName,
            NickName=characterNickName,
            SelfName=characterSelfName
        )
    return string
