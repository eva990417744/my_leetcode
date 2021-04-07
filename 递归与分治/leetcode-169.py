from typing import List


class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        def majority_element_rec(lo: int, hi: int) -> int:
            if lo == hi:
                return nums[lo]
            mid = (hi - lo) // 2 + lo
            left = majority_element_rec(lo, mid)
            right = majority_element_rec(mid + 1, hi)
            if left == right:
                return left
            left_count = sum(1 for i in range(lo, hi + 1) if nums[i] == left)
            right_count = sum(1 for i in range(lo, hi + 1) if nums[i] == right)
            return left if left_count > right_count else right

        return majority_element_rec(0, len(nums) - 1)


a = Solution()
print(a.majorityElement([3, 2, 3]))
