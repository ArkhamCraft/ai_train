import random


class Sort:
    def __init__(self, arr_len):
        self.arr = []
        self.arr_len = arr_len
        self.arr_random()

    def arr_random(self):
        for i in range(self.arr_len):
            self.arr.append(random.randint(0, 99))

    def bubble(self):
        arr = self.arr
        for i in range(self.arr_len - 1, 0, -1):
            for j in range(i):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]


if __name__ == '__main__':
    my_sort = Sort(10)
    print(my_sort.arr)
    my_sort.bubble()
    print(my_sort.arr)
