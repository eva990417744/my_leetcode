class Solution:
    def rotate(self, nums, k):
        long = len(nums)
        if k == 0 or len(nums) <= 1 or k % long == 0:
            return
        if k > long:
            k = k % long
        self.reverse(nums, 0, long - 1 - k)
        self.reverse(nums, long - k, long - 1)
        self.reverse(nums, 0, long - 1)
        print(nums)

    def reverse(self, nums, start, end):
        while start < end:
            temp = nums[start]
            nums[start] = nums[end]
            nums[end] = temp
            end -= 1
            start += 1


a = Solution()
a.rotate([1, 2, 3, 4, 5, 6, 7], 3)
