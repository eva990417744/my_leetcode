from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        count = 0
        last = None
        for i in prices:
            if last is None:
                last = i
                continue
            if i > last:
                count += (i - last)
            last = i
        return count


a = Solution()
print(a.maxProfit([7, 1, 5, 3, 6, 4]))
