from script.Core import CacheContorl,GameConfig,TextLoading
from dateutil import relativedelta
import datetime
import time
import random

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

def getRandDayForYear(year:int) -> datetime.datetime.timetuple :
    '''
    随机获取指定年份中一天的日期
    Keyword arguments:
    year -- 年份
    Return arguments:
    time.time -- 随机日期
    '''
    start = datetime.datetime(year,1,1,0,0,0,0).timestamp()
    end = datetime.datetime(year,12,31,23,59,59).timestamp()
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

def getRandDayForDate(startDate:float,endDate:float) -> datetime.datetime.timetuple:
    '''
    随机获取两个日期中的日期
    Keyword arguments:
    startDate -- 开始日期
    endDate -- 结束日期
    Return arguments:
    time.localtime -- 随机日期
    '''
    t = random.uniform(startDate,endDate)
    return datetime.datetime.fromtimestamp(t).timetuple()

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
