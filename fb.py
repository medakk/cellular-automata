import pygame

class Viewer:

    def __init__(self, update_func, display_size):
        self.update_func = update_func
        pygame.init()
        self.display = pygame.display.set_mode(display_size)
    
    def start(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            Z, new_title = self.update_func()
            surf = pygame.surfarray.make_surface(Z)
            self.display.blit(surf, (0, 0))

            pygame.display.set_caption(new_title)
            pygame.display.update()

        pygame.quit()