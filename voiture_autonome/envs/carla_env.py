# envs/carla_env.py

import carla
import random
import numpy as np


class CarlaEnv:
    """
    Environnement CARLA pour V0 : conduite autonome simple (suivi de voie, adaptation de vitesse).
    Action : [throttle, steer] en continu. Observation : par exemple distance au véhicule devant,
    écart par rapport au centre de voie, vitesse, etc.
    """

    def __init__(self, config):
        # Connexion au serveur CARLA
        self.client = carla.Client(
            config.get("host", "127.0.0.1"), config.get("port", 2000)
        )
        self.client.set_timeout(10.0)
        # Charger la scène (town) et réglages initiaux
        self.world = self.client.load_world(config.get("town", "Town01"))
        weather = getattr(carla.WeatherParameters, config.get("weather", "ClearNoon"))
        self.world.set_weather(weather)
        # Initialiser le traffic manager et le mode synchrone
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS simulé
        self.world.apply_settings(settings)
        tm = self.client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        # Préparer les acteurs (ego et autres véhicules)
        self._setup_actors()
        self.reset()

    def _setup_actors(self):
        # Supprimer les anciens acteurs
        for actor in self.world.get_actors():
            if actor.type_id.startswith("vehicle."):
                actor.destroy()
        # Sélectionner les blueprints de véhicules
        blueprint_library = self.world.get_blueprint_library()
        ego_bp = random.choice(
            blueprint_library.filter("vehicle.*")
        )  # véhicule égo aléatoire
        spawn_points = self.world.get_map().get_spawn_points()
        # Spawn du véhicule égo
        ego_spawn = random.choice(spawn_points)
        self.ego = self.world.try_spawn_actor(ego_bp, ego_spawn)
        # Assurer le spectateur sur l'ego
        if self.ego:
            self.world.get_spectator().set_transform(self.ego.get_transform())
        # Spawn d'un véhicule lent en traffic
        slow_bp = random.choice(blueprint_library.filter("vehicle.*"))
        slow_spawn = random.choice(spawn_points)
        self.slow_vehicle = self.world.try_spawn_actor(slow_bp, slow_spawn)
        if self.slow_vehicle:
            self.slow_vehicle.set_autopilot(True)
            # Donner une probabilité élevée d'ignorer les feux pour qu'il reste sur sa voie
            tm = self.client.get_trafficmanager()
            tm.ignore_lights_percentage(self.slow_vehicle, 100)

    def reset(self):
        """
        Réinitialise l’environnement : repositionne les acteurs et retourne l’état initial.
        """
        self._setup_actors()
        # Avancer le simulateur d'un pas pour stabiliser
        self.world.tick()
        return self._get_observation()

    def step(self, action):
        """
        Applique une action sur le véhicule égo puis avance la simulation.
        Action attendue : [throttle, steer], valeurs dans [0,1] pour throttle et [-1,1] pour steer.
        Retourne : obs, reward, done, info.
        """
        if not self.ego:
            return None, 0.0, True, {}
        # Appliquer le contrôle sur le véhicule égo
        control = carla.VehicleControl()
        throttle, steer = action
        control.throttle = float(np.clip(throttle, 0.0, 1.0))
        control.steer = float(np.clip(steer, -1.0, 1.0))
        self.ego.apply_control(control)
        # Faire évoluer la simulation
        self.world.tick()
        # Obtenir la nouvelle observation
        obs = self._get_observation()
        # Calculer la récompense : par exemple on favorise la vitesse et la stabilité de voie
        speed = self._get_speed(self.ego)
        lane_offset = obs.get("lane_offset", 0.0)
        # Exemple de fonction de récompense simple
        reward = speed * 0.1 - abs(lane_offset) * 0.5
        done = False
        info = {"speed": speed, "lane_offset": lane_offset}
        return obs, reward, done, info

    def _get_observation(self):
        """
        Calcule l'état observé par l'agent : on peut utiliser des capteurs ou des informations du simulateur.
        Ici on renvoie par exemple la distance au véhicule devant, l'écart de voie et la vitesse.
        """
        obs = {}
        # Vitesse du véhicule égo
        obs["speed"] = self._get_speed(self.ego)
        # Écart latéral par rapport au centre de la voie (exemple simplifié)
        ego_loc = self.ego.get_transform().location
        ego_wp = self.world.get_map().get_waypoint(ego_loc, project_to_road=True)
        lane_center = ego_wp.transform.location
        obs["lane_offset"] = ego_loc.y - lane_center.y
        # Distance au véhicule lent (projeté)
        if self.slow_vehicle:
            dist = ego_loc.distance(self.slow_vehicle.get_transform().location)
        else:
            dist = float("inf")
        obs["front_distance"] = dist
        return obs

    def _get_speed(self, vehicle):
        """
        Renvoie la vitesse du véhicule en m/s.
        """
        if vehicle:
            vel = vehicle.get_velocity()
            return 3.6 * np.sqrt(
                vel.x**2 + vel.y**2 + vel.z**2
            )  # convertir en km/h par exemple
        return 0.0
