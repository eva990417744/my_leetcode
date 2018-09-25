def maxProfit(prices):
    sum = 0
    if prices is None:
        return sum
    for i in range(len(prices) - 1):
        if prices[i] < prices[i + 1]:
            sum += prices[i + 1] - prices[i]
    return sum


print(maxProfit([7, 6, 4, 3, 1]))
