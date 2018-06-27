from script.Core import GameConfig,CacheContorl,TextLoading,Dictionaries

# 富文本计算
def setRichTextPrint(textMessage,defaultStyle):
    styleNameList = GameConfig.getFontDataList() + TextLoading.getTextData(TextLoading.barListId,'barlist')
    styleIndex = 0
    styleLastIndex = None
    styleMaxIndex = None
    styleList = []
    for i in range(0,len(styleNameList)):
        styleTextHead = '<' + styleNameList[i] + '>'
        if styleTextHead in textMessage:
            styleIndex = 1
        else:
            pass
    if styleIndex == 0:
        for i in range(0,len(textMessage)):
            styleList.append(defaultStyle)
    else:
        for i in range(0,len(textMessage)):
            if textMessage[i] == '<':
                inputTextStyleSize = textMessage.find('>',i) + 1
                inputTextStyle = textMessage[i + 1:inputTextStyleSize - 1]
                styleLastIndex = i
                styleMaxIndex = inputTextStyleSize
                if inputTextStyle[0] == '/':
                    if CacheContorl.textStylePosition['position'] == 1:
                        CacheContorl.outputTextStyle = 'standard'
                        CacheContorl.textStylePosition['position'] = 0
                        CacheContorl.textStyleCache = ['standard']
                    else:
                        CacheContorl.textStylePosition['position'] = CacheContorl.textStylePosition['position'] - 1
                        CacheContorl.outputTextStyle = CacheContorl.textStyleCache[CacheContorl.textStylePosition['position']]
                else:
                    CacheContorl.textStylePosition['position'] = len(CacheContorl.textStyleCache)
                    CacheContorl.textStyleCache.append(inputTextStyle)
                    CacheContorl.outputTextStyle = CacheContorl.textStyleCache[CacheContorl.textStylePosition['position']]
            else:
                if styleLastIndex != None:
                    if i == len(textMessage):
                        CacheContorl.textStylePosition['position'] = 0
                        CacheContorl.outputTextStyle = 'standard'
                        CacheContorl.textStyleCache = ['standard']
                    if i in range(styleLastIndex,styleMaxIndex):
                        pass
                    else:
                        styleList.append(CacheContorl.outputTextStyle)
                else:
                    styleList.append(CacheContorl.outputTextStyle)
    return styleList

# 移除富文本标签
def removeRichCache(string):
    string = str(string)
    string = Dictionaries.handleText(string)
    barlist = TextLoading.getTextData(TextLoading.barListId, 'barlist')
    styleNameList = GameConfig.getFontDataList() + barlist
    for i in range(0, len(styleNameList)):
        styleTextHead = '<' + styleNameList[i] + '>'
        styleTextTail = '</' + styleNameList[i] + '>'
        if styleTextHead in string:
            string = string.replace(styleTextHead, '')
            string = string.replace(styleTextTail, '')
        else:
            pass
    return string