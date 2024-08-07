import random
import bisect
import itertools
import math
import numpy
from scipy import stats
from typing import Dict, List, Tuple


def two_bit_array_to_dict(array: tuple) -> dict:
    """
    将二维数组转换为字典
    Keyword arguments:
    array -- 要转换的二维数组
    """
    return dict(array)


def get_rand_value_for_value_region(value_list: List[float]) -> float:
    """
    以列表中每个元素的值作为权重随机获取一个元素
    Keyword arguments:
    value_list -- 要计算的列表
    Return arguments:
    int -- 获得的元素
    """
    new_list = [key for key in value_list]
    return random.choices(new_list, weights=new_list)[0]


def get_region_list(now_data: Dict[any, int]) -> dict:
    """
    按dict中每个value的值对key进行排序，并计算权重区域列表
    Keyword arguments:
    now_data -- 需要进行计算权重的dict数据
    """
    sort_data = sorted_dict_for_values(now_data)
    return dict(zip(itertools.accumulate(sort_data.values()), sort_data.keys()))


def sorted_dict_for_values(old_dict: Dict[any, int]) -> dict:
    """
    按dict中每个value的值对key进行排序生成新dict
    Keyword arguments:
    old_dict -- 需要进行排序的数据
    """
    return two_bit_array_to_dict(sorted(old_dict.items(), key=lambda x: x[1]))


def get_random_for_weight(data: Dict[any, int]) -> any:
    """
    按权重随机获取dict中的一个key
    Keyword arguments:
    data -- 需要随机获取key的dict数据
    """
    keys = list(data.keys())
    weights = list(data.values())
    if not sum(weights):
        weights = [1 for i in keys]
    return random.choices(keys, weights=weights)[0]


def get_next_value_for_list(now_int: int, int_list: List[int]) -> int:
    """
    获取列表中第一个比指定值大的数
    Keyword arguments:
    now_int -- 作为获取参考的指定数值
    int_list -- 用于取值的列表
    """
    now_id = bisect.bisect_left(int_list, now_int)
    return int_list[now_id]


def get_old_value_for_list(now_int: int, int_list: List[int]) -> int:
    """
    获取列表中第一个比指定值小的数
    Keyword arguments:
    now_int -- 作为获取参考的指定数值
    int_list -- 用于取值的列表
    Return arguments:
    int -- 查询到的值
    """
    now_id = bisect.bisect_right(int_list, now_int)
    return int_list[now_id - 1]


def list_of_groups(init_list: List[any], children_list_len: int) -> List[List[any]]:
    """
    将列表分割为指定长度的列表集合
    Keyword arguments:
    init_list -- 原始列表
    children_list_len -- 指定长度
    Return arguments:
    List[Tuple[any]] -- 新列表
    """
    list_of_groups = zip(*(iter(init_list),) * children_list_len)
    end_list = [list(i) for i in list_of_groups]
    count = len(init_list) % children_list_len
    if count:
        end_list.append(init_list[-count:])
    return end_list


def linear_decreasing_distribution(start: float, end: float) -> float:
    """
    生成线性递减分布的随机数
    根据线性递减分布，在指定的开始和结束值之间生成一个随机数
    Keyword arguments:
    start -- 分布的起始值
    end -- 分布的结束值
    Return arguments:
    float -- 按照线性递减分布生成的随机数
    """
    return start + (end - start) * (1 - numpy.sqrt(numpy.random.random()))

def custom_distribution(min_value: float, max_value: float) -> float:
    """
    按照特定的概率分布在给定的范围内生成随机数
    这个分布在[min_value, max_value]范围内，中间20%的区域使用贝塔分布，其余部分线性递减
    Keyword arguments:
    min_value -- 最小值
    max_value -- 最大值
    Return arguments:
    float -- 按照指定分布生成的随机数
    """
    mid_range_width = (max_value - min_value) * 0.2
    mid_start = min_value + (max_value - min_value) * 0.4
    mid_end = mid_start + mid_range_width
    rand = numpy.random.random()
    if rand < 0.1:
        return linear_decreasing_distribution(mid_start, min_value)
    elif rand < 0.9:
        return mid_start + stats.beta.rvs(2, 2) * mid_range_width
    else:
        return linear_decreasing_distribution(mid_end, max_value)

