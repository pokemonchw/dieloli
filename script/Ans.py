import core.PyCmd as pycmd
import core.game as game
import script.TextLoading as text
import core.EraPrint as eprint

yesorno = ['4','5','12']
sex = ['6','7','8','9','10']

def option(inputI):
    for i in range(0,len(inputI)):
        id = int(inputI[i])
        pycmd.pcmd(text.loadCmdAdv(inputI[i]),id,None)
        eprint.p('\n')
    ans = game.askfor_int()
    return ans