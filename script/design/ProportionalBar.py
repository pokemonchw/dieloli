# 通用用计算比例条的函数
def getProportionalBar(valueName,maxValue,value,barid):
    proportion = int(int(value)/int(maxValue)* 20)
    trueBar = "1"
    nullBar = "0"
    proportionBar = trueBar * proportion
    fixProportionBar =  nullBar * (20 - proportion)
    proportionBar = '<' + barid + '>' + proportionBar + fixProportionBar + '</' + barid + '>'
    proportionBar = str(valueName) + '[' + proportionBar + ']' + '(' + str(value) + '/' + str(maxValue) + ')'
    return proportionBar

# 通用用于计数条的函数
def getCountBar(valueName,maxValue,value,barid):
    trueBar = "1"
    nullBar = "0"
    countBar = trueBar * int(value)
    fixCountBar = nullBar * (int(maxValue) - int(value))
    countBar = '<' + barid + '>' + countBar + fixCountBar + '</' + barid + '>'
    countBar = str(valueName) + '[' + countBar + ']'
    return countBar