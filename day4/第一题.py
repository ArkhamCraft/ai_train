"""
1.有7个整数，其中有3个数出现了两次，1个数出现了一次， 找出出现了一次的那个数。
"""
nums = [1, 1, 2, 2, 3, 4, 4]
for num in nums:
    if nums.count(num) == 1:
        print(num)

"""
2.写一个简单的for循环，从1打印到20，横着打为1排
"""
for i in range(1, 21):
    print(f'{i} ', end="\t")
print()
"""
3.写一个say_hello函数打印多次hello并给该函数加备注（具体打印几次依靠传递的参数），然后调用say_hello
"""


def say_hello(n):
    for a in range(n):
        print("hello")


n = int(input("输入一个数字:"))
say_hello(n)


"""
4.写一个模块（命名不要用中文），模块里写3个打印函数，然后另外一个py文件调用该模块，并调用对应模块的函数，同时用一下下面操作
"""
import printmod

printmod.dome1()
printmod.dome2()
printmod.dome3()


"""
5.有8个整数，其中有3个数出现了两次，2个数出现了一次， 找出出现了一次的那2个数。
"""
num_list = [11, 11, 22, 22, 33, 44, 44, 55]
for l in num_list:
    if num_list.count(l) == 1:
        print(l)