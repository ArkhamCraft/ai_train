class Women:

    def __init__(self, name, age):
        self.name = name
        self.__age = age

    def __secret(self):
            print(self.__age)

    def boy_friend(self):
        self.__secret()


xiaohong = Women('小红', 18)
xiaohong.boy_friend()
print(xiaohong._Women__age)