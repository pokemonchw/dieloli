from script.Core import CacheContorl,GameConfig,TextLoading
from script.Design import Update
from dateutil import relativedelta
import datetime

# 时间初始化
def initTime():
    CacheContorl.gameTime  = {
        "year":GameConfig.year,
        "month":GameConfig.month,
        "day":GameConfig.day,
        "hour":GameConfig.hour,
        "minute":GameConfig.minute
    }

# 获取时间信息文本
def getDateText(gameTimeData = None):
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

# 获取星期文本
def getWeekDayText():
    weekDay = getWeekDate()
    weekDateData = TextLoading.getTextData(TextLoading.messagePath,'19')
    return weekDateData[int(weekDay)]

# 时间增量
def subTimeNow(minute=0,hour=0,day=0,month=0,year=0):
    newDate = getSubDate(minute,hour,day,month,year)
    CacheContorl.gameTime['year'] = str(newDate.year)
    CacheContorl.gameTime['month'] = str(newDate.month)
    CacheContorl.gameTime['day'] = str(newDate.day)
    CacheContorl.gameTime['hour'] = str(newDate.hour)
    CacheContorl.gameTime['minute'] = str(newDate.minute)

# 获取新日期
def getSubDate(minute=0,hour=0,day=0,month=0,year=0):
    oldDate = "{0}-{1}-{2},{3}:{4}".format(CacheContorl.gameTime['year'],CacheContorl.gameTime['month'],CacheContorl.gameTime['day'],CacheContorl.gameTime['hour'],CacheContorl.gameTime['minute'])
    oldDate = datetime.datetime.strptime(oldDate,"%Y-%m-%d,%H:%M")
    newDate = oldDate + relativedelta.relativedelta(years=year, months=month, days=day, hours=hour, minutes=minute)
    return newDate

def getWeekDate():
    return datetime.datetime(int(CacheContorl.gameTime['year']),int(CacheContorl.gameTime['month']),int(CacheContorl.gameTime['day'])).strftime("%w")

# 获取当前时间段
def getNowTimeSlice():
    pass
