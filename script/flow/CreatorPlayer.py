import core.EraPrint as eprint
import script.TextLoading as textload
import core.game as game

def creator_func():
    eprint.pl(textload.loadMessageAdv(textload.advInputPlayerName))
    playerName = game.askfor_str()
    print(playerName)
    eprint.pl(textload.loadMessageAdv(textload.advEnterPlayerName))
    pass