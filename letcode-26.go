package main

func removeDuplicates(nums []int) int {
	f:=0
	for i:=1;i<len(nums);i++{
		if nums[f]!=nums[i] {
			if i-f>1{
				nums[f+1]=nums[i]
			}
			f++
		}

	}
	return f+1
}
