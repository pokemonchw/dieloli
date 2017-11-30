import core.EraPrint as eprint
import script.TextLoading as textload
import core.game as game
import core.CacheContorl as cache
import script.Ans as ans
import core.PyCmd as pycmd
import core.winframe as winframe

def inputName_func():
    playerId = '0'
    cache.playObject['objectId'] = playerId
    eprint.pl(textload.loadMessageAdv('3'))
    playerName = game.askfor_str()
    cache.temporaryObject = cache.temporaryObjectBak
    cache.temporaryObject['Name'] = playerName
    cache.playObject['object'][playerId] = cache.temporaryObject
    eprint.p(playerName)
    eprint.pnextscreen()
    eprint.pl(textload.loadMessageAdv('4'))
    yrn = ans.option(ans.yesorno)
    if yrn == 4:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputSex_func()
        return
    elif yrn ==5:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputName_func()
    else:
        cache.flowContorl['restartGame'] = 1
        winframe.send_input()
    pass

def inputSex_func():
    ans.option(ans.sex)
    pass