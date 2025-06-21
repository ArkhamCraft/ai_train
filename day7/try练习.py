def duicheng(num):
    num_str = str(num)
    return num_str == num_str[::-1]


try:
    num = int(input("请输入一个整型数："))
    if not duicheng(num):
        raise ValueError("该数不是对数")
    print(f'{num}是一个对数')
except ValueError as e:
    if 'invalid literal for int()' in str(e):
        print("输入不是整型数")

