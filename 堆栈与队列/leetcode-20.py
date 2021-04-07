class Solution:
    def isValid(self, s: str) -> bool:
        stack = []

        for i in s:
            if i in ['(', '{', '[', ]:
                stack.append(i)
            elif i == ')':
                if len(stack) == 0:
                    return False
                if stack.pop() != '(':
                    return False
            elif i == '}':
                if len(stack) == 0:
                    return False
                if stack.pop() != '{':
                    return False
            elif i == ']':
                if len(stack) == 0:
                    return False
                if stack.pop() != '[':
                    return False
        if len(stack) != 0:
            return False
        return True
