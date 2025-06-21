import pygame

pygame.init()
clock = pygame.time.Clock()
i=0
while True:
    clock.tick(60)
    i+=1
    print(i)




pygame.quit()