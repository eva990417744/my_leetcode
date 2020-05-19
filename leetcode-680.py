class Solution:
    def validPalindrome(self, s: str) -> bool:
        def checkPalindrome(low, high):
            i, j = low, high
            while i < j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1
            return True

        low, high = 0, len(s) - 1
        while low < high:
            if s[low] == s[high]:
                low += 1
                high -= 1
            else:
                return checkPalindrome(low + 1, high) or checkPalindrome(low, high - 1)
        return True


class Solution:
    def validPalindrome(self, s: str) -> bool:
        l = len(s)

        if l <= 2:
            return True

        if s == s[::-1]:
            return True

        for i in range(int(l / 2)):
            if s[i] != s[l - 1 - i]:
                tmp1 = s[i:l - 1 - i]
                tmp2 = s[i + 1:l - i]
                break
        if tmp1 == tmp1[::-1] or tmp2 == tmp2[::-1]:
            return True
        return False
