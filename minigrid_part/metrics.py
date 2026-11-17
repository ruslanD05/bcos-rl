import numpy as np
import gymnasium as gym

class ExplanationMetrics(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.offsets = [(0, 0), (7, 0), (0, 7), (7, 7)]

    @property
    def _base(self):
        return self.env.unwrapped

    def get_localization_mask(self):
        base = self._base
        grid_size = base.width
        mask = np.zeros((grid_size, grid_size), dtype=bool)
        active_idx = base.active_idx
        off_x, off_y = self.offsets[active_idx]
        mask[off_y : off_y + 8, off_x : off_x + 8] = True
        return mask

    def get_explanation_quality_mask(self):
        base = self._base
        grid_size = base.width
        mask = np.zeros((grid_size, grid_size), dtype=bool)
        active_idx = base.active_idx
        off_x, off_y = self.offsets[active_idx]
        agent_x, agent_y = base.agent_pos
        mask[agent_y, agent_x] = True
        for x in range(off_x, off_x + 8):
            for y in range(off_y, off_y + 8):
                cell = base.grid.get(x, y)
                if cell is not None and cell.type in ['key', 'door', 'goal']:
                    mask[y, x] = True
        return mask

    def calculate_positive_attribution_ratio(self, attribution_map, mask):
        pos_attr = np.maximum(0, attribution_map)
        total_pos_attr = np.sum(pos_attr)
        if total_pos_attr == 0:
            return 0.0
        return np.sum(pos_attr[mask]) / total_pos_attr