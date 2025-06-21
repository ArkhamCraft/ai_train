class HouseItem:
    def __init__(self, name, area):
        self.name = name
        self.area = area

    def __str__(self):
        return f'{self.name} 占地 {self.area}'


class House:
    def __init__(self, house_type, area):
        self.house_type = house_type
        self.area = area
        self.free_area = area
        self.item_list = []

    def __str__(self):
        return f'户型：{self.house_type} 总面积：{self.area} 剩余：{self.free_area} 家具:{self.item_list}'

    def add_item(self, item: HouseItem) -> None:
        print(f'要添加 {item}')
        if item.area > self.free_area:
            print(f'{item.name} 的面积太大，不能添加到房子')
            return
        self.item_list.append(item.name)
        self.free_area -= item.area


if __name__ == '__main__':
    bed = HouseItem('席梦思', 4)
    chest = HouseItem('衣柜', 2)
    table = HouseItem('餐桌', 1.5)
    print(chest)
    house = House('两室一厅', 60)
    house.add_item(bed)
    print(house)
