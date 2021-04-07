from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dict1 = {}
        for i, k in enumerate(nums):
            if target - k in dict1:
                return [dict1[target - k], i]
            dict1.update({k: i})
        return []
