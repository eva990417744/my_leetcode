def removeDuplicates(nums):
    n = len(nums)
    if n <= 1:
        return n
    index = 0
    for i in range(n - 1):
        if nums[i] != nums[i + 1]:
            nums[index + 1] = nums[i + 1]
            index += 1
    print(nums)
    return index + 1


print(removeDuplicates([0, 0, 1, 1, 1, 2, 2, 3, 3, 4]))
