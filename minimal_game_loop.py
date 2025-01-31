import pygame
import sys

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Minimal Event Test")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            print("Mouse clicked at:", pygame.mouse.get_pos())
    screen.fill((220, 220, 220))  # Fill the background with light gray.
    pygame.display.flip()
    pygame.time.wait(100)

pygame.quit()
sys.exit()
