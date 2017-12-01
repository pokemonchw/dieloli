import core.EraPrint as eprint
import script.TextLoading as textload
import core.game as game
import core.CacheContorl as cache
import script.Ans as ans
import core.PyCmd as pycmd
import core.winframe as winframe
import random

def inputName_func():
    playerId = '0'
    cache.playObject['objectId'] = playerId
    eprint.pl(textload.loadMessageAdv('3'))
    playerName = game.askfor_str()
    cache.temporaryObject = cache.temporaryObjectBak
    cache.temporaryObject['Name'] = playerName
    cache.playObject['object'][playerId] = cache.temporaryObject
    eprint.p(playerName)
    eprint.p('\n')
    eprint.p('\n')
    eprint.pl(textload.loadMessageAdv('4'))
    yrn = ans.option(ans.yesorno)
    if yrn == 4:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputNickName_func()
        return
    elif yrn ==5:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputName_func()
    else:
        cache.flowContorl['restartGame'] = 1
        winframe.send_input()
    pass

def inputNickName_func():
    playerId = '0'
    eprint.pl(textload.loadMessageAdv('5'))
    playerNickName = game.askfor_str()
    cache.temporaryObject['NickName'] = playerNickName
    cache.playObject['object'][playerId] = cache.temporaryObject
    eprint.p(playerNickName)
    eprint.p('\n')
    eprint.p('\n')
    eprint.pl(textload.loadMessageAdv('6'))
    yrn = ans.option(ans.yesorno)
    if yrn == 4:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputSex_func()
        return
    elif yrn ==5:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputNickName_func()
    else:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputName_func()
    pass

def inputSex_func():
    playerId = '0'
    eprint.pl(textload.loadMessageAdv('7'))
    yrn = ans.option(ans.sex)
    sex = textload.loadRoleAtrText('Sex')
    sexList = ['6','7','8','9','10','11']
    for i in  range(0,len(sex)):
        if str(yrn) == sexList[i]:
            sexAtr = sex[i]
            cache.temporaryObject['Sex'] = sexAtr
            cache.playObject['object'][playerId] = cache.temporaryObject
            eprint.p(sexAtr)
            eprint.p('\n')
            eprint.p('\n')
            eprint.pl(textload.loadMessageAdv('8')[sex[i]])
    if yrn == 13:
        rand = random.randint(0,len(sex)-1)
        sexAtr = sex[rand]
        cache.temporaryObject['Sex'] = sexAtr
        cache.playObject['object'][playerId] = cache.temporaryObject
        eprint.p(sexAtr)
        eprint.p('\n')
        eprint.p('\n')
        eprint.pl(textload.loadMessageAdv('8')[sexAtr])
    elif yrn == 12:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputNickName_func()
    pycmd.clr_cmd()
    yrn = ans.option(ans.yesorno)
    if yrn == 4:
        pycmd.clr_cmd()
        eprint.p('\n')
        return
    elif yrn ==5:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputSex_func()
    else:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputNickName_func()
    pass