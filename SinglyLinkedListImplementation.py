class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:

    def __init__(self):
        self.head = None

    def print_list(self):
        cur_node = self.head
        while cur_node:
            print(cur_node.data)
            cur_node = cur_node.next

    def append(self, data):
        new_node = Node(data)

        if self.head is None:
            self.head = new_node
            return

        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node

    def sum_lists(self, llist2):
        p = self.head
        q = llist2.head
        sum_lst = LinkedList()

        carry = 0
        while p or q:
            if not p:
                i = 0
            else:
                i = p.data
            if not q:
                j = 0
            else:
                j = q.data
            s = i + j + carry
            if s >= 10:
                carry = 1
                rem = s % 10
                sum_lst.append(rem)
            else:
                carry = 0
                sum_lst.append(s)
            if p:
                p = p.next
            if q:
                q = q.next

        sum_lst.print_list()

list1 = LinkedList()
list1.append(2)
list1.append(4)
list1.append(3)

list2 = LinkedList()
list2.append(5)
list2.append(6)
list2.append(4)

list1.sum_lists(list2)
