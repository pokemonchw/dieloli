from script.Core import GameConfig,CacheContorl,TextLoading,Dictionaries

def setRichTextPrint(textMessage:str,defaultStyle:str) -> list:
    '''
    获取文本的富文本样式列表
    Keyword arguments:
    textMessage -- 原始文本
    defaultStyle -- 无富文本样式时的默认样式
    '''
    styleNameList = GameConfig.getFontDataList() + list(TextLoading.getGameData(TextLoading.barConfigPath).keys())
    styleIndex = 0
    styleLastIndex = None
    styleMaxIndex = None
    styleList = []
    for i in range(0,len(styleNameList)):
        styleTextHead = '<' + styleNameList[i] + '>'
        if styleTextHead in textMessage:
            styleIndex = 1
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
                    if i not in range(styleLastIndex,styleMaxIndex):
                        styleList.append(CacheContorl.outputTextStyle)
                else:
                    styleList.append(CacheContorl.outputTextStyle)
    return styleList

def removeRichCache(string:str) -> str:
    '''
    移除文本中的富文本标签
    Keyword arguments:
    string -- 原始文本
    '''
    string = str(string)
    string = Dictionaries.handleText(string)
    barlist = list(TextLoading.getGameData(TextLoading.barConfigPath).keys())
    styleNameList = GameConfig.getFontDataList() + barlist
    for i in range(0, len(styleNameList)):
        styleTextHead = '<' + styleNameList[i] + '>'
        styleTextTail = '</' + styleNameList[i] + '>'
        if styleTextHead in string:
            string = string.replace(styleTextHead, '')
            string = string.replace(styleTextTail, '')
    return string
