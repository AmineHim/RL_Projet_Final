# envs/carla_env_v1.py

from .carla_env import CarlaEnv


class CarlaEnvV1(CarlaEnv):
    """
    Environnement CARLA pour V1 : inclut les fonctionnalités V0 et la possibilité de changer de voie pour dépasser.
    Hérite de CarlaEnv et ajoute la logique de dépassement.
    """

    def __init__(self, config):
        super().__init__(config)
        # On peut ajouter d'autres capteurs ici si nécessaire

    def step(self, action):
        """
        Action désormais : [throttle, steer, change_lane] où change_lane est un booléen (0 ou 1).
        Si change_lane==1 et qu'un véhicule lent est devant, l'agent peut effectuer un changement de voie.
        """
        throttle, steer, change_lane = action
        # Implémentation simplifiée du changement de voie
        if change_lane > 0.5 and self._is_slow_vehicle_ahead():
            # Par exemple, on ajuste fortement l'angle pour changer de voie
            steer = 1.0  # tourner complètement à gauche, simplification
        # Appel à la version V0 pour le reste du traitement
        return super().step([throttle, steer])

    def _is_slow_vehicle_ahead(self):
        """
        Détecte si le véhicule lent est dans la voie actuelle (par exemple via la distance).
        """
        if not self.slow_vehicle:
            return False
        ego_loc = self.ego.get_transform().location
        slow_loc = self.slow_vehicle.get_transform().location
        dist = ego_loc.distance(slow_loc)
        return dist < 10.0  # si trop proche, considérer devant
