def containsDuplicate(nums):
    if len(nums) != len(list(set(nums))):
        return True
    else:
        return False


containsDuplicate([1, 1, 2, 2, 3, 4])
