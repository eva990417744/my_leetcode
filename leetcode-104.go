package main

// Definition for a binary tree node.
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

type TreeNodeList struct {
	Node  *TreeNode
	Depth int
}

func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	leftHeight := maxDepth(root.Left)
	rightHeight := maxDepth(root.Right)
	return max(leftHeight, rightHeight) + 1
}

func max(x, y int) int {
	if x < y {
		return y
	}
	return x
}
func dfsMaxDepth(root *TreeNode) int {
	var nodeList []TreeNodeList
	if root != nil {
		newTreeNodeList := TreeNodeList{Node: root, Depth: 1}
		nodeList = append(nodeList, newTreeNodeList)
	}
	depth := 0
	for len(nodeList) != 0 {
		newTreeNodeList := nodeList[len(nodeList)-1]
		nodeList = nodeList[:len(nodeList)-1]
		root = newTreeNodeList.Node
		currentDepth := newTreeNodeList.Depth
		if root != nil {
			depth = max(depth, currentDepth)
			newTreeNodeList := TreeNodeList{Node: root.Left, Depth: currentDepth + 1}
			nodeList = append(nodeList, newTreeNodeList)
			newTreeNodeList = TreeNodeList{Node: root.Right, Depth: currentDepth + 1}
			nodeList = append(nodeList, newTreeNodeList)
		}
	}
	return depth
}
