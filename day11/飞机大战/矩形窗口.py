import pygame
pygame.init()
hero_rect = pygame.Rect(100, 500, 120, 125)
print(f'x {hero_rect.x} y {hero_rect.y}')
print(f'width {hero_rect.width} height {hero_rect.height}')
pygame.quit()