from script.Core import CacheContorl

# 转义文本处理
def handleText(string):
    string = characterName(string)
    string = characterNickName(string)
    string = characterSelfName(string)
    return string

# 转义当前访问对象姓名
def characterName(string):
    try:
        characterId = CacheContorl.characterData['characterId']
        charactername = CacheContorl.characterData['character'][characterId]['Name']
        string = string.replace('{Name}', charactername)
        return string
    except KeyError:
        return string

# 转义当前访问对象昵称
def characterNickName(string):
    try:
        characterId = CacheContorl.characterData['characterId']
        charactername = CacheContorl.characterData['character'][characterId]['NickName']
        string = string.replace('{NickName}', charactername)
        return string
    except KeyError:
        return string

# 转义当前访问对象自称
def characterSelfName(string):
    try:
        characterId = CacheContorl.characterData['characterId']
        characterselfname = CacheContorl.characterData['character'][characterId]['SelfName']
        string = string.replace('{SelfName}',characterselfname)
        return string
    except KeyError:
        return string
