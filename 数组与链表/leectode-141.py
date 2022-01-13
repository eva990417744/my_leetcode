import copy
import math
import random
from collections import defaultdict
from typing import List, Any
import collections
import heapq
from functools import reduce
import functools

import re


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None,
                 random: 'Node' = None, child=None, prev=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
        self.random = random
        self.child = child
        self.prev = prev


class MyLinkedList:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.size = 0
        self.head = ListNode(0)

    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        if index > self.size:
            return -1
        pre = self.head
        for i in range(index + 1):
            pre = pre.next
        return pre.val

    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        """
        pre = ListNode(val)
        pre.next = self.head.next
        self.head.next = pre
        self.size += 1

    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        pre = self.head
        while pre.next:
            pre = pre.next
        pre.next = ListNode(val)
        self.size += 1

    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        """
        if index > self.size:
            return
        elif index <= 0:
            self.addAtHead(val)
        elif index == self.size:
            self.addAtTail(val)
        pre = self.head
        for i in range(index):
            pre = pre.next
        tmp = ListNode(val)
        tmp.next = pre.next
        pre.next = tmp
        self.size += 1

    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        if index < 0 or index >= self.size:
            return
        self.size -= 1
        pre = self.head
        for i in range(index):
            pre = pre.next
        pre.next = pre.next.next


def check(mid: int):
    return True


def mut(l: int, r: int):
    while l < r:
        mid = (l + r + 1) >> 1
        if check(mid):
            l = mid
        else:
            r = mid - 1
    return r


def mut1(l: int, r: int):
    while l < r:
        mid = (l + r) >> 1
        if check(mid):
            l = mid
        else:
            r = mid + 1
    return l


class MyQueue:

    def __init__(self):
        """
        Initialize
        your
        data
        structure
        here.
        """
        self.list = []

    def push(self, x: int) -> None:
        """
        Push
        element
        x
        to
        the
        back
        of
        queue.
        """
        self.list.append(x)

    def pop(self) -> int:
        """
        Removes
        the
        element
        from in front
        of
        queue and returns
        that
        element.
        """
        x = self.list[0]
        self.list[:] = self.list[1:]
        return x

    def peek(self) -> int:
        """
        Get
        the
        front
        element.
        """
        return self.list[0]

    def empty(self) -> bool:
        """
        Returns
        whether
        the
        queue is empty.
        """
        return not self.list


class MyHashMap:

    def __init__(self):
        """
        Initialize
        your
        data
        structure
        here.
        """
        self.dic = {}

    def put(self, key: int, value: int) -> None:
        """
        value
        will
        always
        be
        non - negative.
        """
        self.dic[key] = value

    def get(self, key: int) -> int:
        """
        Returns
        the
        value
        to
        which
        the
        specified
        key is mapped, or -1 if this
        map
        contains
        no
        mapping
        for the key
            """
        if key in self.dic:
            return self.dic[key]
        else:
            return -1

    def remove(self, key: int) -> None:
        """
    Removes
    the
    mapping
    of
    the
    specified
    value
    key if this
    map
    contains
    a
    mapping
    for the key
        """
        if key in self.dic:
            del self.dic[key]


class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.is_end = False
        self.child = {}

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        t = self
        for i in word:
            if i not in t.child:
                t.child[i] = Trie()
            t = t.child[i]
        t.is_end = True

    def search_word(self, word: str):
        t = self
        for i in word:
            if i not in t.child:
                return None
            t = t.child[i]
        return t

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        a = self.search_word(word)
        return a is not None and a.is_end

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        return self.search_word(prefix) is not None


class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.mi = [math.inf]
        self.stack = []

    def push(self, val: int) -> None:
        self.mi.append(min(self.mi[-1], val))
        self.stack.append(val)

    def pop(self) -> None:
        self.mi.pop()
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.mi[-1]


class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.ma = []
        self.mi = []

    def addNum(self, num: int) -> None:
        if len(self.ma) == len(self.mi):
            heapq.heappush(self.ma, -num)
            heapq.heappush(self.mi, -heapq.heappop(self.ma))
        else:
            heapq.heappush(self.mi, num)
            heapq.heappush(self.ma, -heapq.heappop(self.mi))

    def findMedian(self) -> float:
        return self.mi[0] if len(self.mi) != len(self.ma) else (self.mi[0] - self.ma[0]) / 2


class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.l = []
        self.d = dict()

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self.d:
            return False
        self.l.append(val)
        self.d[val] = len(self.l) - 1
        return True

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val not in self.d:
            return False
        index = self.d[val]
        self.l[index] = self.l[-1]
        self.d[self.l[index]] = index
        self.l.pop()
        del self.d[val]
        return True

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        return random.choice(self.l)


class Codec:

    def dfs(self, root: TreeNode, res: List):
        if not root:
            res.append('#')
            return
        res.append(str(root.val))
        self.dfs(root.left, res)
        self.dfs(root.right, res)

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        res = []
        self.dfs(root, res)
        return ' '.join(res)

    def creat_tree(self, data):
        val = data.popleft()
        if val == '#':
            return
        node = TreeNode(val)
        node.left = self.creat_tree(data)
        node.right = self.creat_tree(data)
        return node

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        data = collections.deque(data.split())
        head = self.creat_tree(data)
        return head


class DLinkedNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:

    def __init__(self, capacity: int):
        self.head = DLinkedNode()
        self.last = DLinkedNode()
        self.head.next = self.last
        self.last.prev = self.head
        self.max_long = capacity
        self.l = 0
        self.cache = dict()

    def add_to_head(self, node: DLinkedNode):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def remove_node(self, node: DLinkedNode):
        node.next.prev = node.prev
        node.prev.next = node.next

    def move_to_head(self, node: DLinkedNode):
        self.remove_node(node)
        self.add_to_head(node)

    def remove_last(self):
        node = self.last.prev
        self.remove_node(node)
        return node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self.move_to_head(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key not in self.cache:
            node = DLinkedNode(key, value)
            self.cache[key] = node
            self.add_to_head(node)
            self.l += 1
            if self.l > self.max_long:
                node = self.remove_last()
                del self.cache[node.key]
                self.l -= 1
        else:
            node = self.cache[key]
            node.value = value
            self.move_to_head(node)


class Solution:
    def __init__(self, w: List[int]):
        import itertools
        self.pre = list(itertools.accumulate(w))
        self.total = sum(w)

    def pickIndex(self) -> int:
        import bisect
        x = random.randint(1, self.total)
        return bisect.bisect_left(self.pre, x)


class CQueue:

    def __init__(self):
        self.a = []
        self.b = []

    def appendTail(self, value: int) -> None:
        self.a.append(value)

    def deleteHead(self) -> int:
        if self.b:
            return self.b.pop()
        if not self.a:
            return -1
        while self.a:
            self.b.append(self.a.pop())
        return self.b.pop()


class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        m = len(matrix) + 1
        n = 1 if m == 1 else len(matrix[0]) + 1
        self.sums = [[0] * n for i in range(m)]
        sums = self.sums
        for i in range(m - 1):
            for j in range(n - 1):
                sums[i + 1][j + 1] = sums[i + 1][j] + sums[i][j + 1] + matrix[i][j] - sums[i][j]
        print(self.sums)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.sums[row2 + 1][col2 + 1] + self.sums[row1][col1] - self.sums[row2 + 1][col1] - self.sums[row1][
            col2 + 1]


class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = [-1]
        are = 0
        for i in range(len(heights)):
            while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
                height = heights[stack.pop()]
                width = i - stack[-1] - 1
                are = max(are, height * width)
            stack.append(i)
        while stack[-1] != -1:
            height = heights[stack.pop()]
            width = len(heights) - stack[-1] - 1
            are = max(are, height * width)
        return are

    def maximalRectangle(self, matrix: List[str]) -> int:
        if not matrix:
            return 0
        h = [0] * len(matrix[0])
        are = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] != '0':
                    h[j] += 1
                else:
                    h[j] = 0
            are = max(are, self.largestRectangleArea(h))
        return are


class CBTInserter:

    def __init__(self, root: TreeNode):
        self.root = root
        self.que = collections.deque()
        q = collections.deque([root])
        while q:
            n = len(q)
            for i in range(n):
                node = q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
                if not node.right or not node.left:
                    self.que.append(node)

    def insert(self, v: int) -> int:
        new_node = TreeNode(v)
        node = self.que[0]
        if node.left:
            node.right = new_node
            self.que.popleft()
        else:
            node.left = new_node
        self.que.append(new_node)
        return node.val

    def get_root(self) -> TreeNode:
        return self.root


from threading import Lock
import threading


class ZeroEvenOdd:
    def __init__(self, n):
        self.n = n
        self.lock0 = threading.Lock()
        self.lock2 = threading.Lock()
        self.lock1 = threading.Lock()
        self.lock2.acquire()
        self.lock1.acquire()

    def zero(self, printNumber=print) -> None:
        for i in range(1, self.n + 1):
            self.lock0.acquire()
            printNumber(0)
            if i % 2:
                self.lock1.release()
            else:
                self.lock2.release()

    def even(self, printNumber=print) -> None:
        for i in range(1, self.n + 1):
            if i % 2 == 0:
                self.lock2.acquire()
                printNumber(i)
                self.lock0.release()

    def odd(self, printNumber=print) -> None:
        for i in range(1, self.n + 1):
            if i % 2:
                self.lock1.acquire()
                printNumber(i)
                self.lock0.release()


class FooBar:
    def __init__(self, n):
        from threading import Lock
        self.n = n
        self.f = Lock()
        self.b = Lock()
        self.b.acquire()

    def foo(self, printFoo: 'Callable[[], None]') -> None:

        for i in range(self.n):
            # printFoo() outputs "foo". Do not change or remove this line.
            self.f.acquire()
            printFoo()
            self.b.release()

    def bar(self, printBar: 'Callable[[], None]') -> None:

        for i in range(self.n):
            # printBar() outputs "bar". Do not change or remove this line.
            self.b.acquire()
            printBar()
            self.f.release()


import threading


class FizzBuzz:
    def __init__(self, n: int):
        self.n = n
        # 创建4把锁
        self.FizzLock = threading.Lock()
        self.BuzzLock = threading.Lock()
        self.FizzBuzzLock = threading.Lock()
        self.NumberLock = threading.Lock()

        self.FizzLock.acquire()
        self.BuzzLock.acquire()
        self.FizzBuzzLock.acquire()

    # printFizz() outputs "fizz"
    def fizz(self, printFizz: 'Callable[[], None]') -> None:
        for i in range(1, self.n + 1):
            if i % 3 == 0 and i % 5 != 0:
                self.FizzLock.acquire()
                printFizz()
                self.NumberLock.release()

    # printBuzz() outputs "buzz"
    def buzz(self, printBuzz: 'Callable[[], None]') -> None:
        for i in range(1, self.n + 1):
            if i % 3 != 0 and i % 5 == 0:
                self.BuzzLock.acquire()
                printBuzz()
                self.NumberLock.release()

    # printFizzBuzz() outputs "fizzbuzz"
    def fizzbuzz(self, printFizzBuzz: 'Callable[[], None]') -> None:
        for i in range(1, self.n + 1):
            if i % 3 == 0 and i % 5 == 0:
                self.FizzBuzzLock.acquire()
                printFizzBuzz()
                self.NumberLock.release()

    # printNumber(x) outputs "x", where x is an integer.
    def number(self, printNumber: 'Callable[[int], None]') -> None:
        for i in range(1, self.n + 1):
            self.NumberLock.acquire()
            if i % 3 == 0 and i % 5 == 0:
                self.FizzBuzzLock.release()
            elif i % 3 == 0 and i % 5 != 0:
                self.FizzLock.release()
            elif i % 3 != 0 and i % 5 == 0:
                self.BuzzLock.release()
            else:
                printNumber(i)
                self.NumberLock.release()


class Trie:

    def __init__(self):
        self.children = collections.defaultdict(Trie)
        self.word = ''
        self.is_word = False

    def insert(self, word):
        cur = self
        for c in word:
            cur = cur.children[c]
        cur.is_word = True
        self.word = word


class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.heap = [-i for i in nums]
        self.k = k
        heapq.heapify(self.heap)

    def add(self, val: int) -> int:
        heapq.heappush(self.heap, -val)
        return -self.heap[self.k - 1]


class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        nums2 = nums2[:k]
        nums1 = nums1[:k]
        hea = []
        for i in range(len(nums1)):
            for j in range(len(nums2)):
                hea.append([nums1[i] + nums2[j], nums2[j], (nums1[i], nums2[j])])
        heapq.heapify(hea)
        res = []
        for _ in range(k):
            if not hea:
                break
            _, _, x = heapq.heappop(hea)
            res.append([x[0], x[1]])
        return res


class Solution:
    def minSteps(self, n: int) -> int:
        if n <= 1:
            return 0
        k = 1
        a = 1
        while a < n:
            if a * 2 > n:
                a += 1
            else:
                a *= 2
            k += 1
            print(a, k)
        return k


class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.is_end = False
        self.child = {}

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        t = self
        for i in word:
            if i not in t.child:
                t.child[i] = Trie()
            t = t.child[i]
        t.is_end = True

    def search(self, t: 'Trie', word: str):
        for i in word:
            if i not in t.child:
                return False
            t = t.child[i]
        return True

    def search_all(self, word):
        t = self
        res = False
        for i in range(len(word)):
            for key in self.child.keys():
                if key != word[i] and i != len(word) - 1:
                    res |= t.search(self.child[key], word[i + 1:])
            if word[i] not in t.child:
                return res
            t = t.child[word[i]]
        return res


class MagicDictionary(object):
    def __init__(self):
        self.buckets = collections.defaultdict(list)

    def buildDict(self, words):
        for word in words:
            self.buckets[len(word)].append(word)

    def search(self, word):
        return any(sum(a != b for a, b in zip(word, candidate)) == 1 for candidate in self.buckets[len(word)])


class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        if not nums:
            return [[]]
        nums.sort()
        n = len(nums)
        check = [False] * len(nums)
        res = []

        def dfs(tmp: List):
            if len(tmp) == n:
                res.append(tmp[:])
                return
            for i in range(n):
                if check[i]:
                    continue
                if i > 0 and nums[i] == nums[i - 1] and not check[i - 1]:
                    continue
                check[i] = True
                dfs(tmp + [nums[i]])
                check[i] = False

        res = []
        dfs([])
        return res


class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        n = len(candidates)
        check = [False] * n
        res = []

        def dfs(tmp: List, s: int, index):
            if s == target:
                res.append(tmp)
                return
            elif s > target:
                return
            for i in range(index, n):
                if check[i]:
                    continue
                if i > 0 and candidates[i] == candidates[i - 1] and not check[i - 1]:
                    continue
                check[i] = True
                dfs(tmp + [candidates[i]], s + candidates[i], i + 1)
                check[i] = False

        dfs([], 0, 0)
        return res


a = Solution()
a.replaceWords(dictionary=["cat", "bat", "rat"], sentence="the cattle was rattled by the battery")
