
# 通用用计算比例条的函数
def getProportionalBar(valueName,maxValue,value,iconstyle = 'standard',icon = '=',nilicon = '.',nilidonstyle = 'standard'):
    proportion = int(int(value)/int(maxValue)* 20)
    proportionBar = icon * proportion
    fixProportionBar = '<' + nilidonstyle + ">" + (nilicon * (20 - proportion)) + '</' + nilidonstyle + ">"
    proportionBar = proportionBar + fixProportionBar
    proportionBar = valueName + ':[<' + iconstyle + '>' + proportionBar + '</' + iconstyle + '>]' + '(' + value + '/' + maxValue + ')'
    return proportionBar

# 通用用于计数条的函数
def getCountBar(valueName,maxValue,value,iconstyle = 'standard',icon = '=',nilicon = '.',nilidonstyle = 'standard'):
    countBar = '<' + iconstyle + '>' + (icon * int(value)) + '</' + iconstyle + '>'
    fixCountBar = '<' + nilidonstyle + '>' + (nilicon * (int(maxValue) - int(value))) + '</' + nilidonstyle + '>'
    countBar = valueName + '[' + countBar + fixCountBar + ']'
    return countBar