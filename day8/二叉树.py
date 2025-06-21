from collections import deque


class Node:
    def __init__(self, ele=-1, left=None, right=None):
        self.ele = ele
        self.left = left
        self.right = right


class Tree:
    def __init__(self):
        self.root = None
        self.queue = deque()

    def insert(self, ele):
        new_node = Node(ele)
        self.queue.append(new_node)
        if self.root is None:
            self.root = new_node
        else:
            if self.queue[0].left is None:
                self.queue[0].left = new_node
            else:
                self.queue[0].right = new_node
                self.queue.popleft()

    def pre_order(self, node: Node):
        if node:
            print(node.ele, end=" ")
            self.pre_order(node.left)
            self.pre_order(node.right)

    def mid_order(self, node: Node):
        if node:
            self.mid_order(node.left)
            print(node.ele, end=" ")
            self.mid_order(node.right)

    def last_order(self, node: Node):
        if node:
            self.last_order(node.left)
            self.last_order(node.right)
            print(node.ele, end=" ")


    def level_order(self):
        queue = []
        queue.append(self.root)
        while queue:
            node = queue.pop(0)
            print(node.ele, end=" ")
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)


if __name__ == '__main__':
    tree = Tree()
    for i in range(1, 10):
        tree.insert(i)  # 树的结点插入
    tree.pre_order(tree.root)
    print('\n------------------------')
    tree.mid_order(tree.root)
    print('\n------------------------')
    tree.last_order(tree.root)
    print('\n------------------------')
    tree.level_order()
