package main

func singleNumber(nums []int) int {
	onceNumber := 0
	for _, num := range nums {
		onceNumber ^= num
	}
	return onceNumber
}
