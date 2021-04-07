from typing import List
import heapq


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums:
            return []
        window, res = [], []

        for i, x in enumerate(nums):
            # 维护滑动窗口 使其长度小于等于k
            if i >= k and window[0] <= i - k:
                window.pop(0)
            # 把比新加入值小的去除
            while window and nums[window[-1]] <= x:
                window.pop()
            # 加入新窗口
            window.append(i)
            if i >= k - 1:
                res.append(nums[window[0]])
        return res

    def maxSlidingWindow1(self, nums: List[int], k: int) -> List[int]:
        res, heap = [], []
        for i in range(len(nums)):
            heapq.heappush(heap, (-nums[i], i))
            if i + 1 >= k:
                while heap and heap[0][1] < i + 1 - k:
                    heapq.heappop(heap)
                res.append(-heap[0][0])
        return res


a = Solution()

print(a.maxSlidingWindow1([7, 2, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6], 10))
