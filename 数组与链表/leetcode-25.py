# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        dummy = ListNode(0)
        p = dummy
        while True:
            count = k
            stack = []
            tmp = head
            while count:
                stack.append(tmp)
                tmp = tmp.next
                count -= 1
            if count:
                p.next = head
                break
            while stack:
                p.next = stack.pop()
                p = p.next
            p.next = tmp
            head = tmp
        return dummy.next


head = ListNode(0)
a = None
for i in range(10):
    b = ListNode(i)
    if head.next is None:
        head.next = b
    if a is None:
        a = b
    else:
        a.next = b
    a = b
print(head)
a = Solution()
a.reverseKGroup(head, 3)
