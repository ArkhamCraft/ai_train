class Dog():
    def __init__(self, name, color):
        self.name = name
        self.color = color

    def jiao(self):
        print("汪汪！")

    def weiba(self):
        print("摇尾巴")


xiaohuang = Dog('小黄', '黄色')
xiaohuang.jiao()
xiaohuang.weiba()