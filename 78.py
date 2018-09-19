def subsets(nums):
    l = []
    num = len(nums)
    for a in range(num):
        ls = []
        for b in range(a, num):
            ls.append(nums[b])
        l.append(ls)
    return l


print(subsets([1, 2, 3]))
