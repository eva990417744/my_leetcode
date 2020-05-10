from typing import List


class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        s = 0
        for e in range(1, len(nums)):
            if nums[s] != nums[e] and e - s > 1:
                s += 1
                nums[s] = nums[e]
        return s + 1


def removeDuplicates(nums: List[int]) -> int:
    if not nums:
        return 0
    s = 0
    for e in range(1, len(nums)):
        if nums[s] != nums[e]:
            if e - s > 1:
                nums[s + 1] = nums[e]
            s += 1
    return s + 1


print(removeDuplicates([1, 2]))
