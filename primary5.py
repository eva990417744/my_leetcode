def singleNumber(nums):
    res = 0
    for i in nums:
        res ^= i
    return res


def sing(nums):
    return sum(set(nums)) * 2 - sum(nums)


singleNumber([4, 1, 2, 1, 2])
