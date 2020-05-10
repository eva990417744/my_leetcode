package main
func maxProfit(prices []int) int {
	s:=0
	count:=0
	for e:=1;e<len(prices);e++{
		desc:=prices[e]-prices[s]
		if desc>0{
			count+=desc
		}
		s++
	}
	return count
}