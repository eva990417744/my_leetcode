# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class order:
    def __init__(self):
        self.traverse_path = []

    def pre_order(self, root: TreeNode):
        if root:
            self.traverse_path.append(root.val)
            self.pre_order(root.left)
            self.pre_order(root.right)

    def in_order(self, root: TreeNode):
        if root:
            self.in_order(root.left)
            self.traverse_path.append(root.val)
            self.in_order(root.right)

    def post_order(self, root: TreeNode):
        if root:
            self.post_order(root.left)
            self.post_order(root.right)
            self.traverse_path.append(root.val)


class Solution:
    def in_order(self, root: TreeNode) -> bool:
        if root is None:
            return True
        if not self.in_order(root.left):
            return False
        if self.prev and self.prev.val >= root.val:
            return False
        self.prev = root
        return self.in_order(root.right)

    def isValidBST(self, root: TreeNode) -> bool:
        self.prev = None
        return self.in_order(root)
