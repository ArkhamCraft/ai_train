def bsreach(arr, pos):
    i, j = 0, len(arr)-1
    while i <= j:
        mid = (j + i) // 2
        if arr[mid] < pos:
            i = mid + 1
        elif arr[mid] > pos:
            j = mid - 1
        else:
            return mid
    return -1


# 定义哈希表的最大键值范围（0 ~ MAXKEY-1）
MAXKEY = 1000


def elf_hash(hash_str: str) -> int:
    """
    ELF哈希算法实现（常用于编译器/链接器的符号表处理）
    将输入字符串转换为0~MAXKEY-1范围内的整数哈希值

    参数:
        hash_str: 要计算哈希值的输入字符串

    返回:
        计算得到的哈希值（0 ~ MAXKEY-1）
    """
    h = 0  # 初始化哈希值
    g = 0  # 临时变量，用于处理溢出

    # 遍历字符串中的每个字符
    for i in hash_str:
        # 核心哈希计算步骤：
        # 1. 将当前哈希值左移4位（相当于×16）
        # 2. 加上当前字符的ASCII码值
        h = (h << 4) + ord(i)

        # 检查高4位是否溢出（0xf0000000对应二进制11110000...0000）
        g = h & 0xf0000000

        # 如果高4位有值（g != 0），需要处理溢出
        if g:
            # 将高4位右移24位后与哈希值异或（增加随机性）
            h ^= g >> 24
            # 清除高4位（防止后续计算溢出）
            h &= ~g

    # 返回哈希值对MAXKEY取模，确保在0~MAXKEY-1范围内
    return h % MAXKEY



if __name__ == '__main__':
    my_list = [3, 22, 22, 35, 39, 47, 55, 68, 78, 98]
    print(bsreach(my_list, 39))
    str_list = ["xiongda", "lele", "hanmeimei", "wangdao", "fenghua"]
    hash_table = [None] * MAXKEY  # 初始化一个哈希表
    for i in str_list:
        if hash_table[elf_hash(i)] is None:
            hash_table[elf_hash(i)] = [i]  # 第一次放入
        else:
            hash_table[elf_hash(i)].append(i)  # 哈希冲突后拉链法解决
    print(hash_table)
