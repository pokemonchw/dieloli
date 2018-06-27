from script.Core import CacheContorl,GameConfig,TextLoading

# 时间初始化
def initTime():
    CacheContorl.gameTime['year'] = int(GameConfig.year)
    CacheContorl.gameTime['month'] = int(GameConfig.month)
    CacheContorl.gameTime['day'] = int(GameConfig.day)
    CacheContorl.gameTime['hour'] = int(GameConfig.hour)
    CacheContorl.gameTime['minute'] = int(GameConfig.minute)
    setSubMinute(0)
    pass

# 获取时间信息文本
def getDateText(gameTimeData = None):
    if gameTimeData == None:
        gameTimeData = CacheContorl.gameTime
    else:
        pass
    dateText = TextLoading.getTextData(TextLoading.stageWordId,'65')
    gameYear = str(gameTimeData['year'])
    gameMonth = str(gameTimeData['month'])
    gameDay = str(gameTimeData['day'])
    gameHour = str(gameTimeData['hour'])
    gameMinute = str(gameTimeData['minute'])
    gameYearText = gameYear + TextLoading.getTextData(TextLoading.stageWordId,'59')
    gameMonthText = gameMonth + TextLoading.getTextData(TextLoading.stageWordId,'60')
    gameDayText = gameDay + TextLoading.getTextData(TextLoading.stageWordId,'61')
    gameHourText = gameHour + TextLoading.getTextData(TextLoading.stageWordId,'62')
    gameMinuteText = gameMinute + TextLoading.getTextData(TextLoading.stageWordId,'63')
    dateText = dateText + gameYearText + gameMonthText + gameDayText + gameHourText + gameMinuteText
    return dateText

# 获取星期文本
def getWeekDayText():
    weekDay = getWeekDate()
    weekDateData = TextLoading.getTextData(TextLoading.messageId,'19')
    return weekDateData[weekDay]

# 增加分钟
def setSubMinute(subMinute):
    cacheMinute = CacheContorl.gameTime['minute']
    cacheMinute = int(cacheMinute) + int(subMinute)
    if cacheMinute >= 60:
        subHour = cacheMinute // 60
        cacheMinute = cacheMinute % 60
        CacheContorl.gameTime['minute'] = cacheMinute
        setSubHour(subHour)
    else:
        CacheContorl.gameTime['minute'] = cacheMinute

# 增加小时
def setSubHour(subHour):
    cacheHour = CacheContorl.gameTime['hour']
    cacheHour = int(cacheHour) + int(subHour)
    if cacheHour >= 24:
        subDay = cacheHour // 24
        cacheHour = cacheHour % 24
        CacheContorl.gameTime['hour'] = cacheHour
        setSubDay(subDay)
    else:
        CacheContorl.gameTime['hour'] = cacheHour

# 增加天数
def setSubDay(subDay):
    cacheDay = CacheContorl.gameTime['day']
    cacheMonth = CacheContorl.gameTime['month']
    cacheMonth = int(cacheMonth)
    cacheDay = int(cacheDay) + int(subDay)
    if cacheMonth == 1 or 3 or 5 or 7 or 8 or 10 or 12:
        if cacheDay >= 31:
            setSubMonth("1")
            if cacheDay // 31 > 0:
                CacheContorl.gameTime['day'] = cacheDay - 31
                setSubDay("0")
            else:
                CacheContorl.gameTime['day'] = cacheDay
        else:
            CacheContorl.gameTime['day'] = cacheDay
    elif cacheMonth == 4 or 6 or 9 or 11:
        if cacheDay >= 30:
            setSubMonth("1")
            if cacheDay // 30 > 0:
                CacheContorl.gameTime['day'] = cacheDay - 30
                setSubDay("0")
            else:
                CacheContorl.gameTime['day'] = cacheDay
        else:
            CacheContorl.gameTime['day'] = cacheDay
    elif cacheMonth == 2:
        leapYear = judgeLeapYear()
        if leapYear == "1":
            if cacheDay > 29:
                setSubMonth("1")
                CacheContorl.gameTime['day'] = cacheDay - 29
                if cacheDay // 29 > 0:
                    setSubDay("0")
            else:
                CacheContorl.gameTime['day'] = cacheDay
        else:
            if cacheDay > 28:
                setSubMonth("1")
                CacheContorl.gameTime['day'] = cacheDay - 28
                if cacheDay // 28 > 0:
                    setSubDay("0")
            else:
                CacheContorl.gameTime['day'] = cacheDay

#增加月数
def setSubMonth(subMonth):
    cacheMonth = CacheContorl.gameTime['month']
    cacheMonth = int(cacheMonth) + int(subMonth)
    if cacheMonth > 12:
        subYear = cacheMonth // 12
        cacheMonth = cacheMonth % 12
        CacheContorl.gameTime['month'] = cacheMonth
        setSubYear(subYear)
    else:
        CacheContorl.gameTime['month'] = cacheMonth

# 增加年数
def setSubYear(subMonth):
    cacheYear = CacheContorl.gameTime['year']
    cacheYear = int(cacheYear) + int(subMonth)
    CacheContorl.gameTime['year'] = cacheYear

# 计算星期
def getWeekDate():
    gameTime = CacheContorl.gameTime
    cacheYear = int(gameTime['year'])
    cacheYear = cacheYear - int(cacheYear / 100) * 100
    cacheCentury = int(cacheYear/100)
    cacheMonth = int(gameTime['month'])
    if cacheMonth == 1 or cacheMonth == 2:
        cacheMonth = cacheMonth + 12
        if cacheYear == 0:
            cacheYear = 99
            cacheCentury = cacheCentury - 1
        else:
            cacheYear = cacheYear - 1
    cacheDay =int(gameTime['day'])
    week = cacheYear + int(cacheYear/4) + int(cacheCentury/4) - 2 * cacheCentury + int(26 * (cacheMonth + 1)/10) + cacheDay - 1
    if week < 0:
        weekDay = (week % 7 + 7) % 7
    else:
        weekDay = week % 7
    return weekDay

# 判断当前是否是润年
def judgeLeapYear():
    cacheYear = int(CacheContorl.gameTime['year'])
    if cacheYear % 3200 == 0:
        if cacheYear % 172800 == 0:
            return "1"
        else:
            return "0"
    else:
        if cacheYear % 4 == 0 and cacheYear % 100 != 0:
            return "1"
        elif cacheYear % 100 == 0 and cacheYear % 400 == 0:
            return "1"
        else:
            return "0"
    pass

# 获取当前时间段
def getNowTime():
    pass