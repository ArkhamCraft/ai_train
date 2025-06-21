#列表的增删改查
my_list = [10, 20, 30, 40, 50]
my_list.append(60)
print(my_list)
my_list.insert(2, 20)
print(my_list)
my_list.extend([70, 80])
print(my_list)
my_list.remove(30)
print(my_list)
del my_list[0]
print(my_list)
popped = my_list.pop(2)
print(popped, my_list)
index = my_list.index(50)
print(f"50的下标为：{index}")
print("20是否在list：", 20 in my_list)
print("35是否在list:", 35 in my_list)
count = my_list.count(20)
print("20的出现次数为:",count)
my_list[1] = 25
print(my_list)
my_list[1:3] = [61, 81, 36]
print(my_list)


#字典的增删改查
my_dict = {'name': 'Alice', 'age': 25, 'city': 'New York'}
my_dict['email'] = 'alice@example.com'
print(my_dict)
my_dict['age'] = 30
print(my_dict)
my_dict.update(({'city': 'Boston', 'gender': 'female'}))
print(my_dict)
del my_dict['email']
print(my_dict)
age = my_dict.pop('age')
print((age, my_dict))
print(f"姓名：{my_dict['name']}")
print(my_dict.keys())
print(my_dict.values())
print(my_dict.items())


#字符串的切片
num_str = "0123456789"
cut1 = num_str[2:6]
print(cut1)
cut2 = num_str[2:]
print(cut2)
cut3 = num_str[:6]
print(cut3)
cut4 = num_str[:]
print(cut4)
cut5 = num_str[::2]
print(cut5)
cut6 = num_str[1::2]
print(cut6)
cut7 = num_str[2:-2]
print(cut7)
cut8 = num_str[-2:]
print(cut8)
cut9 = num_str[::-1]
print(cut9)


#使用zip组合下面两个元组，变为列表嵌套元组格式
a = (1, 2, 3)
b = ('a', 'b', 'c')
result = list(zip(a, b))
print(result)


#使用enumerate
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
season_dict = {season: index for index, season in enumerate(seasons)}
print(season_dict)


#求两个有序数字列表的公共元素
list1 = [1, 3, 4, 5, 7, 9]
list2 = [2, 3, 4, 7, 8, 10]


def find_common(lista, listb):
    len1, len2 = len(lista), len(listb)
    i, j = 0, 0
    common = []
    while i < len1 and j < len2:
        if lista[i] == listb[j]:
            common.append(lista[i])
            i += 1
            j += 1
        elif lista[i] < listb[j]:
            i += 1
        else:
            j += 1
    return common


print(find_common(list1, list2))


#给定一个n个整型元素的列表a，其中有一个元素出现次数超过n / 2，求这个元素
my_list = [1, 3, 5, 6, 8, 8, 8, 8, 8, 2]


def major_element(list):
    num = None
    count = 0
    for i in list:
        if count == 0:
            num = i
            count = 1
        elif num == i:
            count += 1
        else:
            count -= 1
    return num


print(major_element(my_list))


#将元组 (1,2,3) 和集合 {4,5,6} 合并成一个列表
a = (1, 2, 3)
b = {4, 5, 6}
c = list(a) + list(b)
print(c)


#在列表 [1,2,3,4,5,6] 首尾分别添加整型元素 7 和 0
my_list = [1, 2, 3, 4, 5, 6]
my_list.insert(0, 7)
my_list.append(0)
print(my_list)


#反转列表 [0,1,2,3,4,5,6,7]后给出中元素 5 的索引号
my_list = [1, 2, 3, 4, 5, 6, 7]
my_list.reverse()
print(my_list)
print(my_list.index(5))


#分别统计列表 [True,False,0,1,2] 中 True,False,0,1,2的元素个数
my_list = [True, False, 0, 1, 2]
print(my_list.count(True))
print(my_list.count(False))
print(my_list.count(0))
print(my_list.count(1))
print(my_list.count(2))
print("True的值为1，False的值为0")


#从列表 [True,1,0,‘x’,None,‘x’,False,2,True] 中删除元素‘x’
my_list = [True, 1, 0, 'x', None, 'x', False, 2, True]
while 'x' in my_list:
    my_list.remove('x')
print(my_list)


#从列表 [True,1,0,‘x’,None,‘x’,False,2,True] 中删除索引号为4的元素
my_list = [True, 1, 0, 'x', None, 'x', False, 2, True]
del my_list[4]
print(my_list)


#删除列表中索引号为奇数（或偶数）的元素
my_list = [True, 1, 0, 'x', None, 'x', False, 2, True]
my_list = [my_list[i] for i in range(len(my_list)) if i % 2 != 0]
print(my_list)


#清空列表中的所有元素
my_list = [True, 1, 0, 'x', None, 'x', False, 2, True]
my_list.clear()
print(my_list)



#对列表 [3,0,8,5,7] 分别做升序和降序排列
my_list = [3, 0, 8, 5, 7]
print(sorted(my_list))
my_list.sort(reverse=True)
print(my_list)


#将列表 [3,0,8,5,7] 中大于 5 元素置为1，其余元素置为0。
my_list = [3, 0, 8, 5, 7]
my_list = [1 if x > 5 else 0 for x in my_list]
print(my_list)


#遍历列表 [‘x’,‘y’,‘z’]，打印每一个元素及其对应的索引号。
my_list = ['x', 'y', 'z']
for i in my_list:
    print(f'{i}--{my_list.index(i)}')


#将列表 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 拆分为奇数组和偶数组两个列表。
my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
list1 = [x for x in my_list if x % 2 != 0]
list2 = [x for x in my_list if x % 2 == 0]
print(list1)
print(list2)


#分别根据每一行的首元素和尾元素大小对二维列表 [[6, 5], [3, 7], [2, 8]] 排序。相当于按6,3,2进行排序，除非第一个元素相等，按第二个元素排序。
my_list = [[6, 5], [3, 7], [2, 8]]
my_list.sort(key=lambda x: (x[0], x[1]))
print(my_list)


#从列表 [1,4,7,2,5,8] 索引为3的位置开始，依次插入列表 [‘x’,‘y’,‘z’] 的所有元素。
list1 = [1, 4, 7, 2, 5, 8]
list2 = ['x', 'y', 'z']
list1[3:3] =list2
print(list1)



#快速生成由 [5,50) 区间内的整数组成的列表
my_list = [x for x in range(5, 50)]
print(my_list)


#将列表 [‘x’,‘y’,‘z’] 和 [1,2,3] 转成 [(‘x’,1),(‘y’,2),(‘z’,3)] 的形式
list1 = ['x', 'y', 'z']
list2 = [1, 2, 3]
my_list = list(zip(list1, list2))
print(my_list)


# 以列表形式返回字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21} 中所有的键
my_dict = {'Alice': 20, 'Beth': 18, 'Cecil': 21}
my_list = list(my_dict.keys())
print(my_list)


#以列表形式返回字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21} 中所有的值
my_dict = {'Alice': 20, 'Beth': 18, 'Cecil': 21}
my_list = list(my_dict.values())
print(my_list)


#以列表形式返回字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21} 中所有键值对组成的元组
my_dict = {'Alice': 20, 'Beth': 18, 'Cecil': 21}
my_list = list(my_dict.items())
print(my_list)


#向字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21} 中追加 ‘David’:19 键值对，更新Cecil的值为17
my_dict = {'Alice': 20, 'Beth': 18, 'Cecil': 21}
my_dict['David'] = 19
my_dict['Cecil'] = 17
print(my_dict)


#删除字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21} 中的Beth键后，清空该字典
my_dict = {'Alice': 20, 'Beth': 18, 'Cecil': 21}
del my_dict['Beth']
print(my_dict)
my_dict.clear()
print(my_dict)


#判断 David 和 Alice 是否在字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21} 中
my_dict = {'Alice': 20, 'Beth': 18, 'Cecil': 21}
print(f'David是否在字典:', 'David' in my_dict)
print(f'Alice是否在字典:', 'Alice' in my_dict)


#遍历字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21}，打印键值对
my_dict = {'Alice': 20, 'Beth': 18, 'Cecil': 21}
for key, value in my_dict.items():
    print(f'{key}:{value}')


#以列表 [‘A’,‘B’,‘C’,‘D’,‘E’,‘F’,‘G’,‘H’] 中的每一个元素为键，默认值都是0，创建一个字典
my_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
my_dict = {key: 0 for key in my_list}
print(my_dict)


#将二维结构 [[‘a’,1],[‘b’,2]] 和 ((‘x’,3),(‘y’,4)) 转成字典
my_list = [['a', 1], ['b', 2]]
my_tuple = (('x', 3), ('y', 4))
my_dict1 = {k: v for k, v in my_list}
my_dict2 = {k: v for k, v in my_tuple}
print(my_dict1, my_dict2)


#将元组 (1,2) 和 (3,4) 合并成一个元组
tuple1 = (1, 2)
tuple2 = (3, 4)
new_tuple = tuple1 + tuple2
print(new_tuple)


#将空间坐标元组 (1,2,3) 的三个元素解包对应到变量 x,y,z
my_tuple = (1, 2, 3)
x, y ,z = my_tuple
print(f'x = {x}, y = {y}, z = {z}')


#返回元组 (‘Alice’,‘Beth’,‘Cecil’) 中 ‘Cecil’ 元素的索引号
my_tuple = ('Alice', 'Beth', 'Cecil')
print(my_tuple.index('Cecil'))


#返回元组 (2,5,3,2,4) 中元素 2 的个数
my_tuple = (2, 5, 3, 2, 4)
print(my_tuple.count(2))


#判断 ‘Cecil’ 是否在元组 (‘Alice’,‘Beth’,‘Cecil’) 中
my_tuple = ('Alice', 'Beth', 'Cecil')
print(f'‘Cecil是否在元组中：', 'Cecil' in my_tuple)


#返回在元组 (2,5,3,7) 索引号为2的位置插入元素 9 之后的新元组
my_tuple = (2, 5, 3, 7)
new_tuple = tuple(my_tuple[:2] + (9, ) + my_tuple[2:])
print(new_tuple)


#创建一个空集合，增加 {‘x’,‘y’,‘z’} 三个元素
my_set = set()
my_set.add('x')
my_set.add('y')
my_set.add('z')
print(my_set)
#删除集合 {‘x’,‘y’,‘z’} 中的 ‘z’ 元素，增j加元素 ‘w’，然后清空整个集合
my_set.discard('z')
print(my_set)
my_set.add('w')
print(my_set)
my_set.clear()
print(my_set)


#返回集合 {‘A’,‘D’,‘B’} 中未出现在集合 {‘D’,‘E’,‘C’} 中的元素（差集）,并集,交集,未重复的元素的集合,是否有重复元素
set1 = {'A', 'D', 'B'}
set2 = {'D', 'E', 'C'}
print(set1.difference(set2))
print(set1.union(set2))
print(set1.intersection(set2))
print(set1.symmetric_difference(set2))
print(set1.isdisjoint(set2))


#判断集合 {‘A’,‘C’} 是否是集合 {‘D’,‘C’,‘E’,‘A’} 的子集
set1 = {'A', 'C'}
set2 = {'D', 'C', 'E', 'A'}
print(set1.issubset(set2))


#去除数组 [1,2,5,2,3,4,5,‘x’,4,‘x’] 中的重复元素
my_list = [1, 2, 5, 2, 3, 4, 5, 'x', 4, 'x']
my_list =list(set(my_list))
print(my_list)



#返回字符串 ‘abCdEfg’ 的全部大写、全部小写和大下写互换形式
my_str ='abCdEfg'
print(my_str.upper())
print(my_str.lower())
print(my_str.swapcase())


#判断字符串 ‘abCdEfg’ 是否首字母大写，字母是否全部小写，字母是否全部大写
my_str ='abCdEfg'
print(my_str.istitle())
print(my_str.islower())
print(my_str.isupper())


#返回字符串 ‘this is python’ 首字母大写以及字符串内每个单词首字母大写形式
my_str = 'this is python'
print(my_str.capitalize())
print(my_str.title())


#判断字符串 ‘this is python’ 是否以 ‘this’ 开头，又是否以 ‘python’ 结尾
my_str = 'this is python'
print(my_str.startswith('this'))
print(my_str.endswith('python'))


#返回字符串 ‘this is python’ 中 ‘is’ 的出现次数
my_str = 'this is python'
print(my_str.count('is'))


#返回字符串 ‘this is python’ 中 ‘is’ 首次出现和最后一次出现的位置
my_str = 'this is python'
print(my_str.find('is'))
print(my_str.rfind('is'))


#将字符串 ‘this is python’ 切片成3个单词
my_str = 'this is python'
print(my_str.split())


#返回字符串 ‘blog.csdn.net/xufive/article/details/102946961’ 按路径分隔符切片的结果
my_str = 'blog.csdn.net/xufive/article/details/102946961'
print(my_str.split('/'))


#将字符串 ‘2.72, 5, 7, 3.14’ 以半角逗号切片后，再将各个元素转成浮点型或整形
my_str = '2.72, 5, 7, 3.14'
strnum = my_str.split(',')
print(strnum)
nums = []
for num in strnum:
    if '.' in num:
        nums.append(float(num))
    else:
        nums.append(int(num))
print(nums)


#判断字符串 ‘adS12K56’ 是否完全为字母数字，是否全为数字，是否全为字母
my_str = 'adS12K56'
print(my_str.isalnum())
print(my_str.isdigit())
print(my_str.isalpha())


#将字符串 ‘there is python’ 中的 ‘is’ 替换为 ‘are’
my_str = 'this is python'
new_str = my_str.replace('is', 'are')
print(new_str)


#清除字符串 ‘\t python \n’ 左侧、右侧，以及左右两侧的空白字符
my_str = '\t python \n'
lnew_str = my_str.lstrip()
print(lnew_str)
rnew_str = my_str.rstrip()
print(rnew_str)
new_str = my_str.strip()
print(new_str)


#将三个全英文字符串（比如，‘ok’, ‘hello’, ‘thank you’）分行打印，实现左对齐、右对齐和居中对齐效果
words = ['ok', 'hello', 'thank you']
width = 15
for word in words:
    print(word.ljust(width))
for word in words:
    print(word.rjust(width))
for word in words:
    print(word.center(width))


#将三个字符串 ‘15’, ‘127’, ‘65535’ 左侧补0成同样长度
my_str = ['15', '127', '65535']
max_len = max(len(s) for s in my_str)
my_str = [s.rjust(max_len, '0') for s in my_str]
print(my_str)


#将列表 [‘a’,‘b’,‘c’] 中各个元素用’|'连接成一个字符串
my_str = ['a', 'b', 'c']
new_str = '|'.join(my_str)
print(new_str)


#将字符串 ‘abc’ 相邻的两个字母之间加上半角逗号，生成新的字符串
my_str = 'abc'
new_str = ','.join(my_str)
print(new_str)


#从键盘输入手机号码，输出形如 ‘Mobile: 186 6677 7788’ 的字符串
pnumber = input("输入手机号码:")
phone_number = f'{pnumber[:3]} {pnumber[3:7]} {pnumber[7:]}'
print(f'Mobile: {phone_number}')


#从键盘输入年月日时分秒，输出形如 ‘2019-05-01 12:00:00’ 的字符串
date = input("输入年月日时分秒:")
date_format = f'{date[:4]}-{date[4:6]}-{date[6:8]} {date[8:10]}:{date[10:12]}:{date[12:]}'
print(date_format)


#给定两个浮点数 3.1415926 和 2.7182818，格式化输出字符串 ‘pi = 3.1416, e = 2.7183’
a = 3.1415926
b = 2.7182818
formatted_num = f'pi = {a:.4f}, e = {b:.4f}'
print(formatted_num)


#将 0.00774592 和 356800000 格式化输出为科学计数法字符串
a = 0.00774592
b = 356800000
formatted_num = f'pi = {a:.4e}, e = {b:.4e}'
print(formatted_num)


#将列表 [0,1,2,3.14,‘x’,None,’’,list(),{5}] 中各个元素转为布尔型
my_list = [0, 1, 2, 3.14, 'x', None, '', list(), {5}]
bool_list = [bool(i) for i in my_list]
print(bool_list)


#返回字符 ‘a’ 和 ‘A’ 的ASCII编码值
a = 'a'
b = 'A'
print(ord(a))
print(ord(b))


#返回ASCII编码值为 57 和 122 的字符
print(chr(57))
print(chr(122))


#将列表 [3,‘a’,5.2,4,{},9,[]] 中 大于3的整数或浮点数置为1，其余置为0
my_list = [3, 'a', 5.2, 4, {}, 9, []]
my_list = [1 if isinstance(i, (int, float)) and i > 3 else 0 for i in my_list]
print(my_list)


#将二维列表 [[1], [‘a’,‘b’], [2.3, 4.5, 6.7]] 转为 一维列表
my_str = [[1], ['a', 'b'], [2.3, 4.5, 6.7]]
new_str = sum(my_str, [])
print(new_str)


#将等长的键列表和值列表转为字典
keys = ['a', 'b', 'c']
values = [1, 2, 3]
my_dict = dict(zip(keys, values))
print(my_dict)



#数字列表求和
a = [51, 61, 98, 34]
total = sum(a)
print(total)