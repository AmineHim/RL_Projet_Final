# utils/reward_utils.py


def compute_speed_reward(speed, target_speed):
    """
    Récompense pour la vitesse : plus proche de target_speed vaut mieux.
    """
    return -abs(speed - target_speed)


def compute_lane_reward(lane_offset):
    """
    Récompense pour l'écart de voie : pénalise l'écart.
    """
    return -abs(lane_offset)
