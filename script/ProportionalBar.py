
def getProportionalBar(valueName,maxValue,value,style = 'standard',icon = '=',nilicon = '.'):
    proportion = int(int(value)/int(maxValue)* 20)
    proportionBar = icon * proportion
    fixProportionBar = nilicon * (20 - proportion)
    proportionBar = proportionBar + fixProportionBar
    proportionBar = valueName + ':[<' + style + '>' + proportionBar + '</' + style + '>]' + '(' + value + '/' + maxValue + ')'
    return proportionBar