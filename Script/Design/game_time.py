import datetime
import time
import random
import bisect
import pysnooper
from dateutil import relativedelta
from Script.Core import cache_contorl, game_config, text_loading


def init_time():
    """
    初始化游戏时间
    """
    cache_contorl.game_time = {
        "year": game_config.year,
        "month": game_config.month,
        "day": game_config.day,
        "hour": game_config.hour,
        "minute": game_config.minute,
    }


def get_date_text(game_time_data=None) -> str:
    """
    获取时间信息描述文本
    Keyword arguments:
    game_timeData -- 时间数据，若为None，则获取当前游戏时间
    """
    if game_time_data == None:
        game_time_data = cache_contorl.game_time
    date_text = text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "65")
    game_year = str(game_time_data["year"])
    game_month = str(game_time_data["month"])
    game_day = str(game_time_data["day"])
    game_hour = str(game_time_data["hour"])
    game_minute = str(game_time_data["minute"])
    game_year_text = game_year + text_loading.get_text_data(
        text_loading.STAGE_WORD_PATH, "59"
    )
    game_month_text = game_month + text_loading.get_text_data(
        text_loading.STAGE_WORD_PATH, "60"
    )
    game_day_text = game_day + text_loading.get_text_data(
        text_loading.STAGE_WORD_PATH, "61"
    )
    game_hour_text = game_hour + text_loading.get_text_data(
        text_loading.STAGE_WORD_PATH, "62"
    )
    game_minute_text = game_minute + text_loading.get_text_data(
        text_loading.STAGE_WORD_PATH, "63"
    )
    date_text = (
        date_text
        + game_year_text
        + game_month_text
        + game_day_text
        + game_hour_text
        + game_minute_text
    )
    return date_text


def get_week_day_text() -> str:
    """
    获取星期描述文本
    """
    week_day = get_week_date()
    week_date_data = text_loading.get_text_data(text_loading.MESSAGE_PATH, "19")
    return week_date_data[int(week_day)]


def sub_time_now(minute=0, hour=0, day=0, month=0, year=0) -> datetime.datetime:
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
    cache_contorl.game_time["year"] = new_date.year
    cache_contorl.game_time["month"] = new_date.month
    cache_contorl.game_time["day"] = new_date.day
    cache_contorl.game_time["hour"] = new_date.hour
    cache_contorl.game_time["minute"] = new_date.minute


def get_sub_date(
    minute=0, hour=0, day=0, month=0, year=0, old_date=None
) -> datetime.datetime:
    """
    获取旧日期增加指定时间后得到的新日期
    Keyword arguments:
    minute -- 增加分钟
    hour -- 增加小时
    day -- 增加天数
    month -- 增加月数
    year -- 增加年数
    old_date -- 旧日期，若为None，则获取当前游戏时间
    """
    if old_date == None:
        old_date = datetime.datetime(
            int(cache_contorl.game_time["year"]),
            int(cache_contorl.game_time["month"]),
            int(cache_contorl.game_time["day"]),
            int(cache_contorl.game_time["hour"]),
            int(cache_contorl.game_time["minute"]),
        )
    new_date = old_date + relativedelta.relativedelta(
        years=year, months=month, days=day, hours=hour, minutes=minute
    )
    return new_date


def get_week_date() -> int:
    """
    计算当前游戏时间属于周几
    Return arguments:
    week_day -- 当前星期数
    """
    return int(timetuple_to_datetime(
        game_time_to_time_tuple(cache_contorl.game_time)
    ).strftime("%w"))


def get_rand_day_for_year(year: int) -> datetime.datetime:
    """
    随机获取指定年份中一天的日期
    Keyword arguments:
    year -- 年份
    Return arguments:
    time.time -- 随机日期
    """
    start = datetime.datetime(year, 1, 1, 0, 0, 0, 0)
    end = datetime.datetime(year, 12, 31, 23, 59, 59)
    return get_rand_day_for_date(start, end)


def timetuple_to_datetime(t: datetime.datetime.timetuple) -> datetime.datetime:
    """
    将timetulp类型数据转换为datetime类型
    Keyword arguments:
    t -- timetulp类型数据
    Return arguments:
    d -- datetime类型数据
    """
    return datetime.datetime(
        t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec
    )


def get_rand_day_for_date(
    start_date: datetime.datetime, end_date: datetime.datetime
) -> datetime.datetime:
    """
    随机获取两个日期中的日期
    Keyword arguments:
    start_date -- 开始日期
    end_date -- 结束日期
    Return arguments:
    time.localtime -- 随机日期
    """
    sub_day = (end_date - start_date).days
    sub_day = random.randint(0, sub_day)
    return get_sub_date(day=sub_day, old_date=start_date)


def system_time_to_game_time(system_time: datetime.datetime.timetuple):
    """
    系统时间戳转换为游戏时间数据
    Keyword arguments:
    system_time -- 系统时间戳
    Return arguments:
    game_time -- 游戏时间数据
    """
    return {
        "year": system_time.tm_year,
        "month": system_time.tm_mon,
        "day": system_time.tm_mday,
    }


def game_time_to_time_tuple(game_time: dict) -> datetime.datetime.timetuple:
    """
    游戏时间数据转换为系统日期
    Keyword arguments:
    game_time -- 游戏时间数据
    Return arguments:
    datetime -- 系统日期
    """
    return datetime.datetime(
        int(game_time["year"]), int(game_time["month"]), int(game_time["day"])
    ).timetuple()


def count_day_for_time_tuple(
    start_date: datetime.datetime.timetuple, end_date: datetime.datetime.timetuple
) -> int:
    """
    计算两个时间之间经过的天数
    Keyword arguments:
    start_date -- 开始时间
    end_date -- 结束时间
    Return arguments:
    int -- 经过天数
    """
    start_day = timetuple_to_datetime(start_date)
    end_day = timetuple_to_datetime(end_date)
    return (start_day - end_day).days


def get_now_time_slice(character_id: int):
    """
    获取当前时间段
    Keyword arguments:
    character_id -- 角色Id
    """
    if cache_contorl.game_time["month"] in range(1, 7) or cache_contorl.game_time[
        "month"
    ] in range(9, 13):
        course_time_judge = judge_course_time(character_id)


def judge_course_time(character_id: int) -> bool:
    """
    校验当前时间是否是上课时间
    Keyword arguments:
    character_id -- 角色Id
    """
    now_week_day = get_week_date()
    character_age = cache_contorl.character_data["character"][character_id].age
    if character_age in range(7, 19):
        phase = character_age - 7
        if phase <= 5 and now_week_day < 5:
            return cache_contorl.course_time_status["PrimarySchool"]["InCourse"]
        elif phase <= 11 and now_week_day < 6:
            return cache_contorl.course_time_status["JuniorMiddleSchool"]["InCourse"]
        else:
            return cache_contorl.course_time_status["SeniorHighSchool"]["InCourse"]


@pysnooper.snoop()
def init_school_course_time_status():
    """
    按当前时间计算各学校上课状态(当前时间是否是上课时间,计算还有多久上课,多久下课)
    """
    course_status = {"InCourse": 0, "ToCourse": 0, "EndCourse": 0}
    cache_contorl.course_time_status["PrimarySchool"] = course_status.copy()
    cache_contorl.course_time_status["JuniorMiddleSchool"] = course_status.copy()
    cache_contorl.course_time_status["SeniorHighSchool"] = course_status.copy()
    if cache_contorl.game_time["month"] in range(1, 7) or cache_contorl.game_time[
        "month"
    ] in range(9, 13):
        cache_contorl.course_time_status["SeniorHighSchool"] = judge_school_course_time(
            "SeniorHighSchool"
        )
        now_week = get_week_date()
        if now_week < 6:
            cache_contorl.course_time_status[
                "JuniorMiddleSchool"
            ] = judge_school_course_time("JuniorMiddleSchool")
        if now_week < 5:
            cache_contorl.course_time_status[
                "PrimarySchool"
            ] = judge_school_course_time("PrimarySchool")

def judge_school_course_time(school_id: str) -> dict:
    """
    校验当前时间是否是学校上课时间
    Keyword arguments:
    school_id -- 学校Id
    """
    course_status = {"InCourse": 0, "ToCourse": 0, "EndCourse": 0}
    course_time_data = text_loading.get_text_data(
        text_loading.COURSE_SESSION_PATH, school_id
    )
    now_time = cache_contorl.game_time["hour"] * 100 + cache_contorl.game_time["minute"]
    end_time_data = {course_time_data[i][1]: i for i in range(len(course_time_data))}
    now_time_index = bisect.bisect_left(list(end_time_data.keys()), now_time)
    if now_time_index >= len(end_time_data):
        return course_status
    start_time = course_time_data[now_time_index][0]
    end_time = course_time_data[now_time_index][1]
    if now_time < start_time:
        if start_time / 100 != now_time / 100:
            index_time = (start_time / 100 - now_time / 100) * 60
            course_status["ToCourse"] == start_time - (
                start_time / 100 - now_time / 100
            ) * 100 + index_time - now_time
        else:
            course_status["ToCourse"] == start_time - now_time
    else:
        course_status["InCourse"] = 1
        if end_time / 100 != now_time / 100:
            index_time = (end_time / 100 - now_time / 100) * 60
            course_status["EndCourse"] == end_time - (
                end_time / 100 - now_time / 100
            ) * 100 + index_time - now_time
        else:
            course_status["EndCourse"] == end_time - now_time
    return course_status
