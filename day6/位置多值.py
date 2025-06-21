
def demo(num, *args, **kwargs):
    print(num)
    print(args)
    print(kwargs)


def dome1(num=3):
    pass


def study_return():
    return 1, 2, 3


def dict_return(a, b, c):
    print(f'a: {a}, b: {b}, c: {c}')


demo(1, 2, 3, 4, 5, name='小明', age=18, gender=True)
my_dict = {'num': 4}
dome1(**my_dict)
a = study_return()
print(a)
my_dict1 = {'a': 1, 'b': 2, 'c': 3}
dict_return(**my_dict1)
