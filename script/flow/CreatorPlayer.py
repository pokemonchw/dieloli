import core.EraPrint as eprint
import script.TextLoading as textload
import core.game as game
import core.CacheContorl as cache
import script.Ans as ans
import core.PyCmd as pycmd
import core.winframe as winframe
import random

playerId = '0'

def inputName_func():
    cache.playObject['objectId'] = playerId
    cache.playObject['object'][playerId] = cache.temporaryObject
    eprint.pl(textload.loadMessageAdv('4'))
    yrn = ans.option(ans.yesorno)
    if yrn == 4:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputNickName_func()
        return
    elif yrn == 5:
        pycmd.clr_cmd()
        eprint.p('\n')
        eprint.pl(textload.loadMessageAdv('3'))
        playerName = game.askfor_str()
        cache.temporaryObject['Name'] = playerName
        eprint.p('\n')
        pycmd.clr_cmd()
        inputName_func()
    elif yrn == 12:
        cache.flowContorl['restartGame'] = 1
        winframe.send_input()
    else:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputName_func()
    pass

def inputNickName_func():
    cache.playObject['object'][playerId] = cache.temporaryObject
    eprint.pl(textload.loadMessageAdv('6'))
    yrn = ans.option(ans.yesorno)
    if yrn == 4:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputSexConfirm_func()
    elif yrn == 5:
        pycmd.clr_cmd()
        eprint.p('\n')
        eprint.pl(textload.loadMessageAdv('5'))
        playerNickName = game.askfor_str()
        cache.temporaryObject['NickName'] = playerNickName
        eprint.p('\n')
        pycmd.clr_cmd()
        inputNickName_func()
    elif yrn == 12:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputName_func()
    else:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputNickName_func()
    pass

def inputSexConfirm_func():
    pycmd.clr_cmd()
    sexId = cache.playObject['object'][playerId]['Sex']
    eprint.pl(textload.loadMessageAdv('8')[sexId])
    yrn = ans.option(ans.yesorno)
    if yrn == 4:
        pycmd.clr_cmd()
        eprint.p('\n')
        return
    elif yrn == 5:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputSexChoice_func()
    elif yrn == 12:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputNickName_func()
    else:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputSexConfirm_func()
    pass

def inputSexChoice_func():
    pycmd.clr_cmd()
    eprint.pl(textload.loadMessageAdv('7'))
    yrn = ans.option(ans.sex)
    sex = textload.loadRoleAtrText('Sex')
    sexList = ['6', '7', '8', '9', '10', '11']
    for i in range(0, len(sex)):
        if str(yrn) == sexList[i]:
            sexAtr = sex[i]
            cache.temporaryObject['Sex'] = sexAtr
            cache.playObject['object'][playerId] = cache.temporaryObject
            pycmd.clr_cmd()
            eprint.p('\n')
            inputSexConfirm_func()
    if yrn == 13:
        rand = random.randint(0, len(sex) - 1)
        sexAtr = sex[rand]
        cache.temporaryObject['Sex'] = sexAtr
        cache.playObject['object'][playerId] = cache.temporaryObject
        pycmd.clr_cmd()
        eprint.p('\n')
        inputSexConfirm_func()
    elif yrn == 12:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputSexConfirm_func()
    else:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputSexChoice_func()
    pass