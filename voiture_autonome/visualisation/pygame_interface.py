# visualisation/pygame_interface.py

import pygame
import numpy as np


class PygameInterface:
    """
    Interface de visualisation avec Pygame pour afficher la vue caméra CARLA.
    """

    def __init__(self, width, height):
        pygame.init()
        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption("CARLA - Vue Caméra")
        self.width = width
        self.height = height

    def show_image(self, image):
        """
        Affiche l'image (array numpy HxWx3) dans la fenêtre Pygame.
        """
        # Convertir l'image (format (H,W,3) BGR->RGB si nécessaire)
        surf = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        self.display.blit(surf, (0, 0))
        pygame.display.flip()

    def handle_events(self):
        """
        Gère les événements Pygame. Retourne False si on ferme la fenêtre.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True
