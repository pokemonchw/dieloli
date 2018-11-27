from script.Core import TextLoading,EraPrint,PyCmd
from script.Design import CmdButtonQueue

# 查看帮助信息面板
def gameHelpPanel():
    PyCmd.clr_cmd()
    titleInfo = TextLoading.getTextData(TextLoading.stageWordId,'85')
    EraPrint.plt(titleInfo)
    EraPrint.p(TextLoading.getTextData(TextLoading.messageId,'31'))
    EraPrint.p('\n')
    inputs = CmdButtonQueue.optionint(CmdButtonQueue.gamehelp,askfor=False)
    EraPrint.plittleline()
    return inputs
