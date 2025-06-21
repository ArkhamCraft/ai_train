#sorted
my_list1 = [1, 3, 2, 4, 5, 3, 2]
print(sorted(my_list1))
print(my_list1)

#sort
print(my_list1.sort())
my_list1.sort()
print(my_list1)
dict1 = {1: 'D', 3: 'B', 2: 'B', 4: 'E', 5: 'A'}
print(sorted(dict1))


str_list = "This is a test string from Andrew".split()
print(str_list)
print(sorted(str_list))


def compare_function(str1: str):
    return str1.lower()


print(sorted(str_list, key=compare_function))
print(sorted(str_list, key= lambda x: x.lower()))

student_tuples = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10),]
print(sorted(student_tuples, key=lambda x: x[2]))


class Student:

    def __init__(self, name, grade, age):
        self.name = name
        self.grade = grade
        self.age = age

    def __repr__(self):
        return repr((self.name, self.grade, self.age))

student_objects = [
    Student('john', 'A', 15),
    Student('jane', 'B', 12),
    Student('dave', 'B', 10),
]
print(sorted(student_objects, key=lambda student: student.age))
print(sorted(student_objects, key=lambda student: student.grade))
print(sorted(student_objects, key=lambda student: student.name))
print('-'*100)
from operator import itemgetter, attrgetter
print(sorted(student_tuples, key=itemgetter(2)))
print(sorted(student_objects, key=attrgetter('age'),reverse=True))
print('-'*100)
print(sorted(student_tuples, key=itemgetter(1, 2)))
print(sorted(student_objects, key=attrgetter('grade', 'age')))
print(sorted(student_tuples, key=lambda student: (student[1],student[2])))
print('-'*100)
data = [('red', 1), ('blue', 1), ('red', 2), ('blue', 2)]
print(sorted(data, key=itemgetter(0)))

print('-'*100)
mydict = { 'Li'   : ['M',7],
           'Zhang': ['E',2],
           'Wang' : ['P',3],
           'Du'   : ['C',2],
           'Ma'   : ['C',9],
           'Zhe'  : ['H',7] }
print(sorted(mydict.items(), key=lambda x:x[1][1]))

gameresult = [
    { "name":"Bob", "wins":10, "losses":3, "rating":75.00 },
    { "name":"David", "wins":3, "losses":5, "rating":57.00 },
    { "name":"Carol", "wins":4, "losses":5, "rating":57.00 },
    { "name":"Patty", "wins":9, "losses":3, "rating": 71.48 }]
print(sorted(gameresult, key=lambda x:x['rating']))

tuples1=[(3,5),(1,2),(2,4),(3,1),(1,3)]
print(sorted(tuples1,key=lambda x:(x[0],-x[1])))