def reverse(x):
    if -2147483648 >= x or x > 2147483647:
        return 0
    if x < 0:
        x = int('-' + str(x)[::-1][:-1])
    else:
        x = int(str(x)[::-1])
    return x


