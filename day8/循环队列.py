class CircleQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.arr = [0] * max_size
        self.front = 0
        self.rear = 0

    def enqueue(self, element):
        if (self.rear + 1) % self.max_size == self.front:
            print("队列已满")
            return
        self.arr[self.rear] = element
        self.rear = (self.rear + 1) % self.max_size

    def dequeue(self):
        if self.rear == self.front:
            print("队列已空")
            return
        element = self.arr[self.front]
        self.front = (self.front + 1) % self.max_size
        return element


if __name__ == '__main__':
    cq = CircleQueue(6)
    cq.enqueue(1)
    cq.enqueue(2)
    cq.enqueue(3)
    cq.enqueue(4)
    cq.enqueue(5)
    cq.enqueue(6)
    print(cq.dequeue())
    print(cq.dequeue())
    print(cq.dequeue())
    print(cq.dequeue())
    print(cq.dequeue())
    print(cq.dequeue())
