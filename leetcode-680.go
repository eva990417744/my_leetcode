package main

// 双指针一头一尾

func isPalindrome(s []byte) bool {
	l, r := 0, len(s)-1
	for l < r {
		if s[l] != s[r] {
			return false
		}
		l++
		r--
	}
	return true
}
func validPalindrome(s string) bool {
	sb := []byte(s)
	l, r := 0, len(sb)-1
	for l < r {
		if sb[l] != sb[r] {
			return isPalindrome(sb[l:r]) || isPalindrome(sb[l+1:r+1])
		}
		l++
		r--
	}
	return true
}
