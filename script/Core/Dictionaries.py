from script.Core import CacheContorl

'''
对文本中的宏进行转义处理
@string 传入文本对象
'''
def handleText(string):
    string = characterName(string)
    string = characterNickName(string)
    string = characterSelfName(string)
    return string

'''
宏:当前对话角色的名字
@string 传入文本对象
'''
def characterName(string):
    try:
        characterId = CacheContorl.characterData['characterId']
        charactername = CacheContorl.characterData['character'][characterId]['Name']
        return string.replace('{Name}', charactername)
    except KeyError:
        return string

'''
宏:当前对话角色的昵称
@string 传入文本对象
'''
def characterNickName(string):
    try:
        characterId = CacheContorl.characterData['characterId']
        charactername = CacheContorl.characterData['character'][characterId]['NickName']
        return string.replace('{NickName}', charactername)
    except KeyError:
        return string

'''
宏:当前对话角色的自称
@string 传入文本对象
'''
def characterSelfName(string):
    try:
        characterId = CacheContorl.characterData['characterId']
        characterselfname = CacheContorl.characterData['character'][characterId]['SelfName']
        return string.replace('{SelfName}',characterselfname)
    except KeyError:
        return string
