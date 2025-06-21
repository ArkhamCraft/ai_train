# 导入pygame库
import pygame

# 初始化pygame所有模块
pygame.init()

# 创建游戏窗口，宽度480像素，高度700像素
screen = pygame.display.set_mode((480, 700))

# 加载背景图片
bg = pygame.image.load("../images/background.png")
# 将背景图片绘制到窗口的(0,0)位置
screen.blit(bg, (0, 0))
# 更新屏幕显示
pygame.display.update()

# 加载英雄飞机图片
hero = pygame.image.load('../images/me1.png')
# 将英雄飞机绘制到窗口的(200,500)位置
screen.blit(hero, (200, 500))
# 更新屏幕显示
pygame.display.update()

# 创建时钟对象，用于控制游戏帧率
clock = pygame.time.Clock()

# 创建英雄飞机的矩形区域，用于碰撞检测
# 参数：x坐标200，y坐标500，宽度102，高度126
hero_rect = pygame.Rect(200, 500, 102, 126)

# 游戏主循环
while True:
    # 控制游戏帧率为60FPS
    clock.tick(60)

    # 英雄飞机向上移动（y坐标减小）
    hero_rect.y -= 1

    # 如果飞机飞出屏幕顶部，则重置到屏幕底部
    if hero_rect.bottom <= 0:
        hero_rect.y = 700

    # 重新绘制背景（覆盖之前的画面）
    screen.blit(bg, (0, 0))
    # 在新的位置绘制英雄飞机
    screen.blit(hero, hero_rect)
    # 更新屏幕显示
    pygame.display.update()

    # 获取所有事件列表
    event_list = pygame.event.get()
    if event_list:
        # 打印所有事件（调试用）
        print(event_list)
        # 遍历所有事件
        for event in event_list:
            # 如果检测到退出事件（如点击窗口关闭按钮）
            if event.type == pygame.QUIT:
                print("游戏退出")
                # 退出pygame
                pygame.quit()
                # 退出程序
                exit(0)