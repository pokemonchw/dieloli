import datetime
import time
import random
import math
import ephem
import os
from types import FunctionType
from dateutil import relativedelta
from Script.Core import (
    cache_control,
    game_type,
    get_text,
)
from Script.Config import normal_config, game_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
gatech = ephem.Observer()
sun = ephem.Sun()
moon = ephem.Moon()
time_zone = datetime.timezone(datetime.timedelta(hours=+8))
#os.environ["TZ"] = time_zone.__str__()
#time.tzset()


def init_time():
    """
    初始化游戏时间
    """
    game_time = datetime.datetime(
        normal_config.config_normal.year,
        normal_config.config_normal.month,
        normal_config.config_normal.day,
        normal_config.config_normal.hour,
        normal_config.config_normal.minute,
        tzinfo=time_zone,
    )
    game_time = game_time.astimezone(time_zone)
    cache.game_time = game_time.timestamp()


def get_date_text(text_time: int = 0) -> str:
    """
    获取时间信息描述文本
    Keyword arguments:
    text_time -- 时间数据，若为0，则获取当前游戏时间
    """
    if not text_time:
        text_time = cache.game_time
    game_time_data = datetime.datetime.fromtimestamp(text_time, time_zone)
    return _("时间:{year}年{month}月{day}日{hour}点{minute}分").format(
        year=game_time_data.year,
        month=game_time_data.month,
        day=game_time_data.day,
        hour=game_time_data.hour,
        minute=game_time_data.minute,
    )


def get_week_day_text() -> str:
    """
    获取星期描述文本
    """
    now_date = datetime.datetime.fromtimestamp(cache.game_time, time_zone)
    week_day = now_date.weekday()
    week_date_data = game_config.config_week_day[week_day]
    return week_date_data.name


def sub_time_now(minute=0, hour=0, day=0, month=0, year=0):
    """
    增加当前游戏时间
    Keyword arguments:
    minute -- 增加的分钟
    hour -- 增加的小时
    day -- 增加的天数
    month -- 增加的月数
    year -- 增加的年数
    """
    new_date = get_sub_date(minute, hour, day, month, year)
    cache.game_time = new_date


def get_sub_date(
    minute=0,
    hour=0,
    day=0,
    month=0,
    year=0,
    old_date: int = 0,
) -> int:
    """
    获取旧日期增加指定时间后得到的新日期
    Keyword arguments:
    minute -- 增加分钟
    hour -- 增加小时
    day -- 增加天数
    month -- 增加月数
    year -- 增加年数
    old_date -- 旧日期，若为0，则获取当前游戏时间
    """
    if not old_date:
        old_date = cache.game_time
    if old_date > 0:
        old_date_data = datetime.datetime.fromtimestamp(old_date)
    else:
        old_date_data = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=old_date)
    new_date: datetime.datetime = old_date_data + relativedelta.relativedelta(
        years=year, months=month, days=day, hours=hour, minutes=minute
    )
    if new_date.year > 1970:
        return new_date.timestamp()
    return (new_date - datetime.datetime(1970, 1, 1)).total_seconds()


def get_rand_day_for_year(year: int) -> int:
    """
    随机获取指定年份中一天的日期
    Keyword arguments:
    year -- 年份
    Return arguments:
    time.time -- 随机日期
    """
    start = datetime.datetime(year, 1, 1, 0, 0, 0, 0)
    end = datetime.datetime(year, 12, 31, 23, 59, 59)
    if year > 1970:
        return get_rand_day_for_date(start.timestamp(), end.timestamp())
    start_time_stamp = (start - datetime.datetime(1970, 1, 1)).total_seconds()
    end_time_stamp = (end - datetime.datetime(1970, 1, 1)).total_seconds()
    return random.randint(start_time_stamp, end_time_stamp)


def timetuple_to_datetime(t: datetime.datetime.timetuple) -> datetime.datetime:
    """
    将timetulp类型数据转换为datetime类型
    Keyword arguments:
    t -- timetulp类型数据
    Return arguments:
    d -- datetime类型数据
    """
    return datetime.datetime(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


def get_rand_day_for_date(start_date: datetime.datetime, end_date: datetime.datetime) -> int:
    """
    随机获取两个日期中的日期
    Keyword arguments:
    start_date -- 开始日期
    end_date -- 结束日期
    Return arguments:
    time.localtime -- 随机日期
    """
    sub_day = int((end_date - start_date) / 86400)
    sub_day = random.randint(0, sub_day)
    return get_sub_date(day=sub_day, old_date=start_date)


def count_day_for_datetime(
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> int:
    """
    计算两个时间之间经过的天数
    Keyword arguments:
    start_date -- 开始时间
    end_date -- 结束时间
    Return arguments:
    int -- 经过天数
    """
    return (start_date - end_date).days


def judge_date_big_or_small(time_a: int, time_b: int) -> int:
    """
    比较当前时间是否大于或等于旧时间
    Keyword arguments:
    time_a -- 当前时间
    time_b -- 旧时间
    Return arguments:
    0 -- 小于
    1 -- 大于
    2 -- 等于
    """
    if time_a == time_b:
        return 2
    return time_b < time_a


def ecliptic_lon(now_time: int) -> float:
    """
    根据日期计算黄经
    now_time -- 日期
    Return arguments:
    float -- 黄经
    """
    s = ephem.Sun(now_time)
    equ = ephem.Equatorial(s.ra, s.dec, epoch=now_time)
    e = ephem.Ecliptic(equ)
    return e.lon


def get_solar_period(now_time: int) -> int:
    """
    根据日期计算对应节气id
    Keyword arguments:
    now_time -- 日期
    Return arguments:
    int -- 节气id
    """
    e = ecliptic_lon(datetime.datetime.fromtimestamp(now_time))
    n = int(e * 180.0 / math.pi / 15)
    return n


def get_old_solar_period_time(now_time: int) -> (int, int):
    """
    根据日期计算上个节气的开始日期
    Keyword arguments:
    now_time -- 日期
    Return arguments:
    new_time -- 节气日期
    """
    s1 = get_solar_period(now_time)
    s0 = s1
    dt = 1
    new_time = now_time
    while True:
        new_time = get_sub_date(day=-dt, old_date=new_time)
        s = get_solar_period(new_time)
        if s0 != s:
            s0 = s
            dt = -dt / 2
        if s != s1 and abs(dt) < 1:
            break
    return new_time, s0


def get_next_solar_period_time(now_time: int) -> (int, int):
    """
    根据日期计算下个节气的开始日期
    Keyword arguments:
    now_time -- 日期
    Return arguments:
    new_time -- 节气日期
    """
    s1 = get_solar_period(now_time)
    s0 = s1
    dt = 1
    new_time = now_time
    while True:
        new_time = get_sub_date(day=dt, old_date=new_time)
        s = get_solar_period(new_time)
        if s0 != s:
            s0 = s
            dt = -dt / 2
        if s != s1 and abs(dt) < 1:
            break
    return new_time, s0


def judge_datetime_solar_period(now_time: int) -> (bool, int):
    """
    校验日期是否是节气以及获取节气id
    Keyword arguments:
    now_time -- 日期
    Return arguments:
    bool -- 校验结果
    int -- 节气id
    """
    new_time, solar_period = get_old_solar_period_time(now_time)
    new_time = datetime.datetime.fromtimestamp(new_time)
    now_time = datetime.datetime.fromtimestamp(now_time)
    if new_time.year == now_time.year and new_time.month == now_time.month and new_time.day == now_time.day:
        return 1, solar_period
    new_time, solar_period = get_next_solar_period_time(now_time.timestamp())
    new_time = datetime.datetime.fromtimestamp(new_time)
    if new_time.year == now_time.year and new_time.month == now_time.month and new_time.day == now_time.day:
        return 1, solar_period
    return 0, 0


def get_sun_time(old_time: int) -> int:
    """
    根据时间获取太阳位置id
    Keyword arguments:
    old_time -- 时间
    Return arguments:
    int -- 太阳位置id
    """
    if "sun_phase" not in cache.__dict__:
        cache.__dict__["sun_phase"] = {}
    old_time = datetime.datetime.fromtimestamp(old_time, time_zone)
    now_time = old_time.astimezone(time_zone)
    if now_time.hour > old_time.hour:
        now_time = datetime.datetime(
            old_time.year, old_time.month, old_time.day, old_time.hour, old_time.minute, tzinfo=time_zone
        )
    now_date_str = f"{now_time.year}/{now_time.month}/{now_time.day}"
    gatech.long, gatech.lat = str(cache.school_longitude), str(cache.school_latitude)
    if (
        (now_date_str not in cache.sun_phase)
        or (now_time.hour not in cache.sun_phase[now_date_str])
        or (now_time.minute not in cache.sun_phase[now_date_str][now_time.hour])
    ):
        now_unix = now_time.timestamp()
        now_unix -= 60
        for _ in range(0, 1439):
            now_unix += 60
            now_unix_date = datetime.datetime.fromtimestamp(now_unix,time_zone)
            now_unix_date.replace(tzinfo=time_zone)
            new_unix_date = now_unix_date.astimezone(time_zone.utc)
            gatech.date = new_unix_date
            sun.compute(gatech)
            now_az = sun.az * 57.2957795
            now_unix_date_str = f"{now_unix_date.year}/{now_unix_date.month}/{now_unix_date.day}"
            cache.sun_phase.setdefault(now_unix_date_str, {})
            cache.sun_phase[now_unix_date_str].setdefault(now_unix_date.hour, {})
            cache.sun_phase[now_unix_date_str][now_unix_date.hour][now_unix_date.minute] = get_sun_phase_for_sun_az(now_az)
        if len(cache.sun_phase) > 1:
            del_date = sorted(list(cache.sun_phase.keys()))[0]
            if del_date != now_date_str:
                del cache.sun_phase[del_date]
    return cache.sun_phase[now_date_str][now_time.hour][now_time.minute]


def get_sun_phase_for_sun_az(now_az: float) -> int:
    """
    根据太阳夹角获取太阳位置对应配表id
    Keyword arguments:
    now_az -- 太阳夹角
    Return arguments:
    太阳位置配表id
    """
    if 225 <= now_az < 255:
        return 8
    if 255 <= now_az < 285:
        return 9
    if 285 <= now_az < 315:
        return 10
    if 315 <= now_az < 345:
        return 11
    if now_az >= 345 or now_az < 15:
        return 0
    if 15 <= now_az < 45:
        return 1
    if 45 <= now_az < 75:
        return 2
    if 75 <= now_az < 105:
        return 3
    if 105 <= now_az < 135:
        return 4
    if 135 <= now_az < 165:
        return 5
    if 165 <= now_az < 195:
        return 6
    return 7


def get_moon_phase(now_time: int) -> int:
    """
    根据时间获取月相配置id
    Keyword arguments:
    now_time -- 时间
    Return arguments:
    int -- 月相配置id
    """
    if "moon_phase" not in cache.__dict__:
        cache.__dict__["moon_phase"] = {}
    now_time = datetime.datetime.fromtimestamp(now_time)
    now_date_str = f"{now_time.year}/{now_time.month}/{now_time.day}"
    if now_date_str not in cache.moon_phase:
        new_time = datetime.datetime(now_time.year, now_time.month, now_time.day)
        new_time.astimezone(time_zone)
        gatech.date = datetime.datetime.utcfromtimestamp(time.mktime(new_time.utctimetuple()))
        moon.compute(gatech)
        now_phase = moon.phase
        gatech.date += 1
        moon.compute(gatech)
        next_phase = moon.phase
        now_type = next_phase > now_phase
        for phase in game_config.config_moon_data[now_type]:
            phase_config = game_config.config_moon[phase]
            if phase_config.min_phase < now_phase <= phase_config.max_phase:
                cache.moon_phase[now_date_str] = phase_config.cid
                break
        if len(cache.moon_phase) > 3:
            del_date = list(cache.moon_phase.keys())[0]
            del cache.moon_phase[del_date]
    return cache.moon_phase[now_date_str]


def judge_attend_class_today(character_id: int) -> bool:
    """
    校验角色今日是否需要上课
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_time = character_data.behavior.start_time
    if not now_time:
        now_time = cache.game_time
    now_time = datetime.datetime.fromtimestamp(now_time)
    now_week = now_time.weekday()
    now_month = now_time.month
    if now_month not in {3, 4, 5, 6, 8, 9, 10, 11, 12}:
        return 0
    if character_data.age <= 18:
        if character_data.age > 15:
            return 1
        if character_data.age > 13 and now_week < 6:
            return 1
        if now_week < 5:
            return 1
    elif (
        character_id in cache.teacher_phase_table
        and now_week in cache.teacher_class_week_day_data[character_id]
    ):
        return 1
    return 0
