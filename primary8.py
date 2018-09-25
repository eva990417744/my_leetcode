def moveZeroes(nums):
    size, index_s = len(nums), 0
    for i in range(size):
        if nums[i] != 0:
            nums[index_s] = nums[i]
            index_s += 1
    nums[index_s:] = [0] * (size - index_s)
    print(nums)


moveZeroes([0, 1, 0, 3, 12])
