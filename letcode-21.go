package main

//Definition for singly-linked list.
type ListNode struct {
	Val  int
	Next *ListNode
}

func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	prehead := new(ListNode)
	pre := prehead
	for l1 != nil && l2 != nil {
		if l1.Val > l2.Val {
			pre.Next = l2
			l2=l2.Next
		} else {
			pre.Next = l1
			l1=l1.Next
		}
		pre = pre.Next
	}
	if l1 == nil {
		pre.Next = l2
	} else {
		pre.Next = l1
	}
	return prehead.Next
}
