# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if root is None:
            return 0
        left_height = self.maxDepth(root.left)
        right_height = self.maxDepth(root.right)
        return max(left_height, right_height) + 1

    def dfs_max_depth(self, root: TreeNode) -> int:
        node_list = []
        if root:
            node_list.append((1, root))
        depth = 0
        while node_list:
            current_depth, node = node_list.pop()
            if node:
                depth = max(depth, current_depth)
                node_list.append((current_depth + 1, node.left))
                node_list.append((current_depth + 1, node.right))
        return depth
