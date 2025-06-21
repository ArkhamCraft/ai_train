import random


class Sort:
    def __init__(self, arr_len):
        self.arr = []
        self.arr_len = arr_len
        self.arr_random()

    def arr_random(self):
        for i in range(self.arr_len):
            self.arr.append(random.randint(0, 100))

    def qsort(self, low, height):
        arr = self.arr
        k = low
        random_pos = random.randint(low, height)
        arr[random_pos], arr[height] = arr[height], arr[random_pos]
        for i in range(low, height):
            if arr[i] < arr[height]:
                arr[i], arr[k] = arr[k], arr[i]
                k += 1
        arr[k], arr[height] = arr[height], arr[k]
        return k

    def quick_sort(self, low, height):
        if low < height:
            pos = self.qsort(low, height)
            self.quick_sort(low, pos - 1)
            self.quick_sort(pos + 1, height)

    def adjust_max_heap(self, adjust_pos, arr_len):
        arr = self.arr
        dad = adjust_pos
        son = 2 * dad + 1
        while son < arr_len:
            if son + 1 < arr_len and arr[son] < arr[son + 1]:
                son += 1
            if arr[son] > arr[dad]:
                arr[son], arr[dad] = arr[dad], arr[son]
                dad = son
                son = 2 * son + 1
            else:
                break

    def heap_sort(self):
        arr = self.arr
        for i in range(self.arr_len // 2 - 1, -1, -1):
            self.adjust_max_heap(i, self.arr_len)
        for n in range(self.arr_len-1, 0, -1):
            arr[0], arr[n] = arr[n], arr[0]
            self.adjust_max_heap(0, n)


if __name__ == '__main__':
    my_list = Sort(10)
    print(my_list.arr)
    # s.quick_sort(0, len(s.arr) - 1)
    my_list.heap_sort()
    print(my_list.arr)
