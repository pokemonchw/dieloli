from script.Core import CacheContorl,GameConfig,TextLoading

# 时间初始化
def initTime():
    CacheContorl.gameTime['year'] = int(GameConfig.year)
    CacheContorl.gameTime['month'] = int(GameConfig.month)
    CacheContorl.gameTime['day'] = int(GameConfig.day)
    CacheContorl.gameTime['hour'] = int(GameConfig.hour)
    CacheContorl.gameTime['minute'] = int(GameConfig.minute)
    subTimeNow()

# 获取时间信息文本
def getDateText(gameTimeData = None):
    if gameTimeData == None:
        gameTimeData = CacheContorl.gameTime
    else:
        pass
    dateText = TextLoading.getTextData(TextLoading.stageWordPath,'65')
    gameYear = str(gameTimeData['year'])
    gameMonth = str(gameTimeData['month'])
    gameDay = str(gameTimeData['day'])
    gameHour = str(gameTimeData['hour'])
    gameMinute = str(gameTimeData['minute'])
    gameYearText = gameYear + TextLoading.getTextData(TextLoading.stageWordPath,'59')
    gameMonthText = gameMonth + TextLoading.getTextData(TextLoading.stageWordPath,'60')
    gameDayText = gameDay + TextLoading.getTextData(TextLoading.stageWordPath,'61')
    gameHourText = gameHour + TextLoading.getTextData(TextLoading.stageWordPath,'62')
    gameMinuteText = gameMinute + TextLoading.getTextData(TextLoading.stageWordPath,'63')
    dateText = dateText + gameYearText + gameMonthText + gameDayText + gameHourText + gameMinuteText
    return dateText

# 获取星期文本
def getWeekDayText():
    weekDay = getWeekDate()
    weekDateData = TextLoading.getTextData(TextLoading.messagePath,'19')
    return weekDateData[weekDay]

# 时间增量
def subTimeNow(minute = 0,hour = 0,day = 0,month = 0,year = 0):
    subMinute = 0
    nowYear = int(CacheContorl.gameTime['year'])
    subToYear = nowYear + year
    if year != 0:
        subMinute = subMinute + subYearToMinute(nowYear,subToYear)
    if month != 0:
        nowMonth = CacheContorl.gameTime["month"]
        monthSubYear = month // 12
        nowToYear = subToYear + monthSubYear
        if monthSubYear != 0:
            month = month % 12
            subMinute = subMinute + subYearToMinute(subToYear,nowToYear)
        if month != 0:
            for i in range(1,month + 1):
                nowMonth = nowMonth + i
                if nowMonth in [1,2,4,6,8,9,11]:
                    subMinute = subMinute + 31 * 24 * 60
                elif nowMonth == 3:
                    if judgeLeapYear(nowToYear) == '1':
                        subMinute = subMinute + 29 * 24 * 60
                    else:
                        subMinute = subMinute + 28 * 24 * 60
                else:
                    subMinute = subMinute + 30 * 24 * 60
    if day != 0:
        subMinute = subMinute + day * 24 * 60
    if hour != 0:
        subMinute = subMinute + hour * 60
    subMinute = subMinute + minute
    CacheContorl.subGameTime = subMinute
    subTime = CacheContorl.gameTime.copy()
    subTime["minute"] = subTime["minute"] + subMinute
    CacheContorl.gameTime = setSubMinute(subTime)

# 新增年数转换为分钟
def subYearToMinute(nowYear,toYear):
    subMinute = 0
    for i in range(nowYear,toYear):
        if judgeLeapYear(i) == '1':
            subDay = 366
        else:
            subDay = 365
        subMinute = subMinute + subDay * 24 * 60
    return subMinute

# 增加分钟
def setSubMinute(subTime):
    cacheMinute = subTime["minute"]
    if cacheMinute >= 60:
        subHour = cacheMinute // 60
        cacheMinute = cacheMinute % 60
        subTime["hour"] = subHour
        subTime["minute"] = cacheMinute
        return setSubHour(subTime)
    else:
        subTime["minute"] = cacheMinute
        return subTime

# 增加小时
def setSubHour(subTime):
    cacheHour = subTime["hour"]
    if cacheHour >= 24:
        subDay = cacheHour // 24
        cacheHour = cacheHour % 24
        subTime["hour"] = cacheHour
        subTime["day"] = subDay
        return setSubDay(subTime)
    else:
        return subTime

# 增加天数
def setSubDay(subTime):
    cacheMonth = subTime['month']
    cacheDay = subTime["day"]
    if cacheMonth in [1,3,5,7,8,10,12]:
        if cacheDay >= 31:
            subTime["month"] = subTime["month"] + 1
            subTime = setSubMonth(subTime)
            if cacheDay // 31 > 0:
                subTime['day'] = cacheDay - 31
                return setSubDay(subTime)
            else:
                subTime["day"] = cacheDay
                return subTime
        else:
            subTime["day"] = cacheDay
            return subTime
    elif cacheMonth in [4,6,9,11]:
        if cacheDay >= 30:
            subTime["month"] = subTime["month"] + 1
            subTime = setSubMonth(subTime)
            if cacheDay // 30 > 0:
                subTime["day"] = cacheDay - 30
                return setSubDay(subTime)
            else:
                subTime["day"] = cacheDay
                return subTime
        else:
            subTime["day"] = cacheDay
            return subTime
    elif cacheMonth == 2:
        leapYear = judgeNowLeapYear()
        if leapYear == "1":
            if cacheDay > 29:
                subTime["month"] = subTime["month"] + 1
                subTime["day"] = cacheDay - 29
                if cacheDay // 29 > 0:
                    return setSubDay(subTime)
                else:
                    return subTime
            else:
                subTime["day"] = cacheDay
                return subTime
        else:
            if cacheDay > 28:
                setSubMonth("1")
                subTime["month"] = subTime["month"] + 1
                subTime["day"] = cacheDay - 28
                if cacheDay // 28 > 0:
                    return setSubDay(subTime)
                else:
                    return subTime
            else:
                subTime["day"] = cacheDay
                return subTime

#增加月数
def setSubMonth(subTime):
    cacheMonth = CacheContorl.gameTime['month']
    subMonth = subTime["month"]
    cacheMonth = int(cacheMonth) + subMonth
    if cacheMonth > 12:
        subYear = cacheMonth // 12
        cacheMonth = cacheMonth % 12
        subTime['month'] = cacheMonth
        subTime["year"] = subTime["year"] + subYear
        return subTime
    else:
        subTime['month'] = cacheMonth
        return subTime

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
def judgeNowLeapYear():
    cacheYear = int(CacheContorl.gameTime['year'])
    return judgeLeapYear(cacheYear)

def judgeLeapYear(year):
    year = int(year)
    if year % 3200 == 0:
        if year % 172800 == 0:
            return "1"
        else:
            return "0"
    else:
        if year % 4 == 0 and year % 100 != 0:
            return "1"
        elif year % 100 == 0 and year % 400 == 0:
            return "1"
        else:
            return "0"

# 获取当前时间段
def getNowTimeSlice():
    pass
