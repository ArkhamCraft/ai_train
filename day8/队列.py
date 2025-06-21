from collections import deque
my_que = deque(['a', 'b', 'c', 'd'])
my_que.append('e')
print(my_que)
my_que.popleft()
print(my_que)
my_que.pop()
print()
my_que[1] = 'xxx0'
print(my_que)