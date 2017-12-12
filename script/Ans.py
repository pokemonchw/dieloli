import core.PyCmd as pycmd
import core.game as game
import script.TextLoading as text
import core.EraPrint as eprint

logomenu = ['1','2','3']
yesorno = ['4','5','12']
inputNickNameList = ['4','5','14','12']
sex = ['6','7','8','9','10','11','13','12']

def option(inputI):
    for i in range(0,len(inputI)):
        id = int(inputI[i])
        pycmd.pcmd(text.loadCmdAdv(inputI[i]),id,None)
        eprint.p('\n')
    ans = game.askfor_int()
    return ans