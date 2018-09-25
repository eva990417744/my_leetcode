def plusOne(digits):
    if digits is None or len(digits) == 0:
        return [1]
    temp = [str(i) for i in digits]
    number = int("".join(temp))
    return [int(i) for i in list(str(number + 1))]


print(plusOne([9]))
