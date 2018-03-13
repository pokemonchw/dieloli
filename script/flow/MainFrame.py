import core.EraPrint as eprint
import script.GameTime as gametime
import core.TextLoading as textload

# 游戏主页
def mainFrame_func():
    eprint.p('\n')
    titleText = textload.getTextData(textload.stageWordId,'64')
    eprint.plt(titleText)
    dateText = gametime.getDateText()
    eprint.p(dateText)
    eprint.p('\n')