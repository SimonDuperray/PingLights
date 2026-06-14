from abc import ABC, abstractmethod


class StrategyRebond(ABC):
    def __init__(self, seuil_rebond, delai_min_frames, seuil_perte_balle):
        self.seuil_rebond = seuil_rebond
        self.delai_min_frames = delai_min_frames
        self.seuil_perte_balle = seuil_perte_balle

    @abstractmethod
    def on_balle_detectee(self, positions, frame_count, dernier_rebond_frame):
        """Appelé à chaque frame où la balle est détectée. Retourne (rebond_pos, zone) ou None."""
        pass

    @abstractmethod
    def on_balle_perdue(self, positions, frame_count, dernier_rebond_frame, frames_sans_balle):
        """Appelé à chaque frame où la balle n'est pas détectée. Retourne (rebond_pos, zone) ou None."""
        pass

    def _analyser_trajectoire(self, traj, frame_count, dernier_rebond_frame, pixels_vers_cm, get_zone):
        if len(traj) < 3:
            return None

        idx_min = max(range(len(traj)), key=lambda i: traj[i][1])

        if not (1 <= idx_min <= len(traj) - 2):
            return None

        direction_avant = traj[idx_min][1] - traj[idx_min - 1][1]
        direction_apres = traj[idx_min + 1][1] - traj[idx_min][1]

        if direction_avant > self.seuil_rebond and direction_apres < -self.seuil_rebond:
            if frame_count - dernier_rebond_frame > self.delai_min_frames:
                rebond_pos = traj[idx_min]
                x_cm, y_cm = pixels_vers_cm(rebond_pos[0], rebond_pos[1])
                zone = get_zone(x_cm, y_cm)
                return rebond_pos, zone

        return None


class StrategyTempsReel(StrategyRebond):
    """Analyse la trajectoire à chaque nouvelle position détectée."""

    def on_balle_detectee(self, positions, frame_count, dernier_rebond_frame, pixels_vers_cm, get_zone):
        return self._analyser_trajectoire(list(positions), frame_count, dernier_rebond_frame, pixels_vers_cm, get_zone)

    def on_balle_perdue(self, positions, frame_count, dernier_rebond_frame, frames_sans_balle, pixels_vers_cm, get_zone):
        return None


class StrategyFallback(StrategyRebond):
    """Analyse la trajectoire uniquement après la perte de la balle."""

    def on_balle_detectee(self, positions, frame_count, dernier_rebond_frame, pixels_vers_cm, get_zone):
        return None

    def on_balle_perdue(self, positions, frame_count, dernier_rebond_frame, frames_sans_balle, pixels_vers_cm, get_zone):
        if frames_sans_balle == self.seuil_perte_balle:
            return self._analyser_trajectoire(list(positions), frame_count, dernier_rebond_frame, pixels_vers_cm, get_zone)
        return None


class StrategyCombo(StrategyRebond):
    """Temps réel en priorité, fallback si rien détecté."""

    def on_balle_detectee(self, positions, frame_count, dernier_rebond_frame, pixels_vers_cm, get_zone):
        return self._analyser_trajectoire(list(positions), frame_count, dernier_rebond_frame, pixels_vers_cm, get_zone)

    def on_balle_perdue(self, positions, frame_count, dernier_rebond_frame, frames_sans_balle, pixels_vers_cm, get_zone):
        if frames_sans_balle == self.seuil_perte_balle:
            return self._analyser_trajectoire(list(positions), frame_count, dernier_rebond_frame, pixels_vers_cm, get_zone)
        return None
