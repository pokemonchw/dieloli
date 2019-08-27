def getProportionalBar(valueName,maxValue,value,barid):
    '''
    通用用于计算比例条的函数
    Keyword arguments:
    valueName -- 比例条名字
    maxValue -- 最大数值
    value -- 当前数值
    barid -- 用于填充比例条的图形id
    '''
    proportion = int(int(value)/int(maxValue)* 20)
    trueBar = "1"
    nullBar = "0"
    proportionBar = trueBar * proportion
    fixProportionBar =  nullBar * (20 - proportion)
    proportionBar = '<' + barid + '>' + proportionBar + fixProportionBar + '</' + barid + '>'
    proportionBar = str(valueName) + '[' + proportionBar + ']' + '(' + str(value) + '/' + str(maxValue) + ')'
    return proportionBar

def getCountBar(valueName,maxValue,value,barid):
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
