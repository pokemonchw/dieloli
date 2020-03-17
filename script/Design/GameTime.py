from script.Core import CacheContorl,GameConfig,TextLoading
from dateutil import relativedelta
import datetime
import time
import random
import bisect

def initTime():
    '''
    初始化游戏时间
    '''
    CacheContorl.gameTime  = {
        "year":GameConfig.year,
        "month":GameConfig.month,
        "day":GameConfig.day,
        "hour":GameConfig.hour,
        "minute":GameConfig.minute
    }

def getDateText(gameTimeData = None) -> str:
    '''
    获取时间信息描述文本
    Keyword arguments:
    gameTimeData -- 时间数据，若为None，则获取当前CacheContorl.gameTime
    '''
    if gameTimeData == None:
        gameTimeData = CacheContorl.gameTime
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

def getWeekDayText() -> str:
    '''
    获取星期描述文本
    '''
    weekDay = getWeekDate()
    weekDateData = TextLoading.getTextData(TextLoading.messagePath,'19')
    return weekDateData[int(weekDay)]

def subTimeNow(minute=0,hour=0,day=0,month=0,year=0) -> datetime.datetime:
    '''
    增加当前游戏时间
    Keyword arguments:
    minute -- 增加的分钟
    hour -- 增加的小时
    day -- 增加的天数
    month -- 增加的月数
    year -- 增加的年数
    '''
    newDate = getSubDate(minute,hour,day,month,year)
    CacheContorl.gameTime['year'] = newDate.year
    CacheContorl.gameTime['month'] = newDate.month
    CacheContorl.gameTime['day'] = newDate.day
    CacheContorl.gameTime['hour'] = newDate.hour
    CacheContorl.gameTime['minute'] = newDate.minute

def getSubDate(minute=0,hour=0,day=0,month=0,year=0,oldDate=None) -> datetime.datetime:
    '''
    获取旧日期增加指定时间后得到的新日期
    Keyword arguments:
    minute -- 增加分钟
    hour -- 增加小时
    day -- 增加天数
    month -- 增加月数
    year -- 增加年数
    oldDate -- 旧日期，若为None，则获取当前游戏时间
    '''
    if oldDate == None:
        oldDate = datetime.datetime(
            int(CacheContorl.gameTime['year']),
            int(CacheContorl.gameTime['month']),
            int(CacheContorl.gameTime['day']),
            int(CacheContorl.gameTime['hour']),
            int(CacheContorl.gameTime['minute'])
        )
    newDate = oldDate + relativedelta.relativedelta(
        years=year,
        months=month,
        days=day,
        hours=hour,
        minutes=minute
    )
    return newDate

def getWeekDate() -> int:
    '''
    计算当前游戏时间属于周几
    Return arguments:
    weekDay -- 当前星期数
    '''
    return timetupleTodatetime(gameTimeToDatetime(CacheContorl.gameTime)).strftime("%w")

def getRandDayForYear(year:int) -> datetime.datetime :
    '''
    随机获取指定年份中一天的日期
    Keyword arguments:
    year -- 年份
    Return arguments:
    time.time -- 随机日期
    '''
    start = datetime.datetime(year,1,1,0,0,0,0)
    end = datetime.datetime(year,12,31,23,59,59)
    return getRandDayForDate(start,end)

def timetupleTodatetime(t:datetime.datetime.timetuple) -> datetime.datetime:
    '''
    将timetulp类型数据转换为datetime类型
    Keyword arguments:
    t -- timetulp类型数据
    Return arguments:
    d -- datetime类型数据
    '''
    return datetime.datetime(t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec)

def getRandDayForDate(startDate:datetime.datetime,endDate:datetime.datetime) -> datetime.datetime:
    '''
    随机获取两个日期中的日期
    Keyword arguments:
    startDate -- 开始日期
    endDate -- 结束日期
    Return arguments:
    time.localtime -- 随机日期
    '''
    subDay = (endDate - startDate).days
    subDay = random.randint(0,subDay)
    return getSubDate(day=subDay,oldDate=startDate)

def systemTimeToGameTime(systemTime:datetime.datetime.timetuple):
    '''
    系统时间戳转换为游戏时间数据
    Keyword arguments:
    systemTime -- 系统时间戳
    Return arguments:
    gameTime -- 游戏时间数据
    '''
    return {
        'year':systemTime.tm_year,
        'month':systemTime.tm_mon,
        'day':systemTime.tm_mday
    }

def gameTimeToDatetime(gameTime:dict) -> datetime.datetime.timetuple:
    '''
    游戏时间数据转换为系统日期
    Keyword arguments:
    gameTime -- 游戏时间数据
    Return arguments:
    datetime -- 系统日期
    '''
    return datetime.datetime(int(gameTime['year']),int(gameTime['month']),int(gameTime['day'])).timetuple()

def countDayForDateToDate(startDate:datetime.datetime.timetuple,endDate:datetime.datetime.timetuple) -> int:
    '''
    计算两个时间之间经过的天数
    Keyword arguments:
    startDate -- 开始时间
    endDate -- 结束时间
    Return arguments:
    int -- 经过天数
    '''
    startDay = timetupleTodatetime(startDate)
    endDay = timetupleTodatetime(endDate)
    return (startDay - endDay).days

def getNowTimeSlice(characterId:int):
    '''
    获取当前时间段
    Keyword arguments:
    characterId -- 角色Id
    '''
    if CacheContorl.gameTime['Month'] in range(1,7) or CacheContorl.gameTime['Month'] in range(9,13):
        courseTimeJudge = judgeCourseTime(characterId)

def judgeCourseTime(characterId:int) -> bool:
    '''
    校验当前时间是否是上课时间
    Keyword arguments:
    characterId -- 角色Id
    '''
    nowWeekDay = getWeekDate()
    characterAge = CacheContorl.characterData['character'][characterId].Age
    if characterAge in range(7,19):
        phase = characterAge - 7
        if phase <= 5 and nowWeekDay < 5:
            return CacheContorl.courseTimeStatus['PrimarySchool']['InCourse']
        elif phase <= 11 and nowWeekDay < 6:
            return CacheContorl.courseTimeStatus['JuniorMiddleSchool']['InCourse']
        else:
            return CacheContorl.courseTimeStatus['SeniorHighSchool']['InCourse']

def initSchoolCourseTimeStatus():
    '''
    按当前时间计算各学校上课状态(当前时间是否是上课时间,计算还有多久上课,多久下课)
    '''
    courseStatus = {
        "InCourse":0,
        "ToCourse":0,
        "EndCourse":0
    }
    CacheContorl.courseTimeStatus['PrimarySchool'] = courseStatus.copy()
    CacheContorl.courseTimeStatus['JuniorMiddleSchool'] = courseStatus.copy()
    CacheContorl.courseTimeStatus['SeniorHighSchool'] = courseStatus.copy()
    if CacheContorl.gameTime['Month'] in range(1,7) or CacheContorl.gameTime['Month'] in range(9,13):
        CacheContorl.courseTimeStatus['SeniorHighSchool'] = judgeSeniorCourseTime('SeniorHighSchool')
        nowWeek = getWeekDate()
        if nowWeek < 6:
            CacheContorl.courseTimeStatus['JuniorMiddleSchool'] = judgeJuniorCourseTime('JuniorMiddleSchool')
        if nowWeek < 5:
            CacheContorl.courseTimeStatus['PrimarySchool'] = judgePrimaryCourseTime('PrimarySchool')

def judgeSchoolCourseTime(schoolId:str) -> dict:
    '''
    校验当前时间是否是学校上课时间
    Keyword arguments:
    schoolId -- 学校Id
    '''
    courseStatus = {
        "InCourse":0,
        "ToCourse":0,
        "EndCourse":0
    }
    courseTimeData = TextLoading.getTextData(TextLoading.courseSession,schoolId)
    nowTime = CacheContorl.gameTime['Hour'] * 100 + CacheContorl.gameTime['Minute']
    endTimeData = {courseTimeData[i][1]:i for i in range(len(courseTimeData))}
    nowTimeIndex = bisect.bisect_left(endTimeData.keys(),nowTime)
    if nowTimeIndex >= len(endTimeData):
        return courseStatus
    startTime = courseTimeData[nowTimeIndex][0]
    endTime = courseTimeData[nowTimeIndex][1]
    elif nowTime < startTime:
        if startTime / 100 != nowTime / 100:
            indexTime = (startTime / 100 - nowTime / 100) * 60
            courseStatus['ToCourse'] == startTime - (startTime / 100 - nowTime / 100) * 100 + indexTime - nowTime
        else:
            courseStatus['ToCourse'] == startTime - nowTime
    else:
        courseStatus['InCourse'] = 1
        if endTime / 100 != nowTime / 100:
            indexTime = (endTime / 100 - nowTime / 100) * 60
            courseStatus['EndCourse'] == endTime - (endTime / 100 - nowTime / 100) * 100 + indexTime - nowTime
        else:
            courseStatus['EndCourse'] == endTime - nowTime
    return courseStatus
