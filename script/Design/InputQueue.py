from script.Core import GameInit,EraPrint,TextLoading

def waitInput(intA:int,intB:int) -> GameInit.askfor_str:
    '''
    等待玩家输入ab之间的一个数
    Keyword arguments:
    intA -- 输入边界A
    intB -- 输入边界B
    '''
    while(True):
        ans = GameInit.askfor_str()
        if ans.isdecimal():
            ans = int(ans)
            if intA <= ans <= intB:
                break
        EraPrint.pl(ans)
        EraPrint.pl(TextLoading.getTextData(TextLoading.errorPath,'inputNullError') + '\n')
    return ans
