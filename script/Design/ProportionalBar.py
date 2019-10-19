from script.Core import GameConfig,TextHandle

def getProportionalBar(valueName:str,maxValue:int,value:int,barid:str,textWidth=0) -> str:
    '''
    通用用于计算比例条的函数
    Keyword arguments:
    valueName -- 比例条名字
    maxValue -- 最大数值
    value -- 当前数值
    barid -- 用于填充比例条的图形id
    textWidth -- 进度条区域宽度 (default 0)
    '''
    if textWidth == 0:
        textWidth = GameConfig.text_width
    barWidth = textWidth - TextHandle.getTextIndex(valueName) - 5 - TextHandle.getTextIndex(str(maxValue)) - TextHandle.getTextIndex(str(value))
    proportion = int(int(value)/int(maxValue)* barWidth)
    trueBar = "1"
    nullBar = "0"
    proportionBar = trueBar * proportion
    fixProportionBar =  nullBar * int(barWidth - proportion)
    proportionBar = '<' + barid + '>' + proportionBar + fixProportionBar + '</' + barid + '>'
    proportionBar = str(valueName) + '[' + proportionBar + ']' + '(' + str(value) + '/' + str(maxValue) + ')'
    return proportionBar

def getCountBar(valueName:str,maxValue:int,value:int,barid:str) -> str:
    '''
    通用用于计算计数条的函数
    Keyword arguments:
    valueName -- 比例条名字
    maxValue -- 最大数值
    value -- 当前数值
    barid -- 用于填充比例条的图形id
    '''
    trueBar = "1"
    nullBar = "0"
    countBar = trueBar * int(value)
    fixCountBar = nullBar * (int(maxValue) - int(value))
    countBar = '<' + barid + '>' + countBar + fixCountBar + '</' + barid + '>'
    countBar = str(valueName) + '[' + countBar + ']'
    return countBar
