
def getProportionalBar(valueName,maxValue,value):
    proportion = int(int(value)/int(maxValue)* 20)
    proportionBar = '=' * proportion
    fixProportionBar = '.' * (20 - proportion)
    proportionBar = proportionBar + fixProportionBar
    proportionBar = valueName + ':[' + proportionBar + ']' + '(' + value + '/' + maxValue + ')'
    return proportionBar