import core.EraPrint as eprint
import script.TextLoading as textload
import core.game as game
import core.CacheContorl as cache

def creator_func():
    playerId = '0'
    cache.playObject['objectId'] = playerId
    eprint.pl(textload.loadMessageAdv(textload.advInputPlayerName))
    playerName = game.askfor_str()
    cache.temporaryObject = cache.temporaryObjectBak
    cache.temporaryObject['Name'] = playerName
    cache.playObject['object'][playerId] = cache.temporaryObject
    eprint.p(playerName)
    eprint.pnextscreen()
    eprint.pl(textload.loadMessageAdv(textload.advEnterPlayerName))
    pass