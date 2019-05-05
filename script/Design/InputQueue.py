from script.Core import GameInit,EraPrint,TextLoading

# 等待玩家输入ab之间的数
def waitInput(intA,intB):
    while(True):
        ans = GameInit.askfor_str()
        if ans.isdecimal():
            ans = int(ans)
            if intA <= ans <= intB:
                break
        EraPrint.pl(ans)
        EraPrint.pl(TextLoading.getTextData(TextLoading.errorPath,'inputNullError') + '\n')
    return ans
