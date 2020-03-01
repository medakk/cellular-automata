import pygame

class Viewer:
    """
    Represents a pygame backed frame viewer
    """

    def __init__(self, update_func, display_size):
        """
        Parameters
        ----------

        update_func : function
            A function that must return a numpy uint8 array with dimensions (W, H, 3)
        display_size : tuple
            The shape of the display window, in the form of (W, H)
        """

        self.update_func = update_func
        pygame.init()
        self.display = pygame.display.set_mode(display_size)
    
    def set_title(self, title):
        """
        Set the title of the window

        Parameters
        ----------

        title : str
            The new title
        """
        pygame.display.set_caption(title)
    
    def start(self):
        """
        Starts the display loop
        """

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            Z = self.update_func()
            surf = pygame.surfarray.make_surface(Z)
            self.display.blit(surf, (0, 0))

            pygame.display.update()

        pygame.quit()