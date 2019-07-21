from script.Core import CacheContorl

'''
对文本中的宏进行转义处理
@string 传入文本对象
'''
def handleText(string):
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
