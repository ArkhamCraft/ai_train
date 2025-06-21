def count_1(num):
    binary = bin(num)
    nums = binary.count('1')
    return nums
inum = int(input())
print(count_1(inum))