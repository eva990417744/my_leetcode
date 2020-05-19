from typing import List


class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        once_number = 0
        for i in nums:
            once_number ^= i
        return once_number
