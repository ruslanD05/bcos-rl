import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall, Goal, Lava, Door, Key
import random
import matplotlib.pyplot as plt
from minigrid.envs import UnlockEnv

class RandomLavaEnv(MiniGridEnv):
    def __init__(self, size=8, num_lava=9, **kwargs):
        self.num_lava = num_lava
        mission_space = MissionSpace(mission_func=lambda: "reach the goal avoiding lava")
        super().__init__(
            grid_size=size,
            max_steps=4 * size**2,
            mission_space=mission_space,
            **kwargs
        )

    def _gen_grid(self, width, height):
        while True:
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)

            # Place agent and goal randomly
            self.place_agent()
            self.goal_pos = self.place_obj(Goal())

            # Place lava randomly
            for _ in range(self.num_lava):
                self.place_obj(Lava())

            # Check if goal is reachable using Breadth-First Search
            if self._is_reachable(self.agent_pos, self.goal_pos):
                break # Valid grid found

    def _is_reachable(self, start, target):
        queue = [start]
        visited = {start}
        
        while queue:
            x, y = queue.pop(0)
            if (x, y) == target:
                return True
                
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if (nx, ny) not in visited:
                    cell = self.grid.get(nx, ny)
                    # We can walk on empty cells or the goal itself
                    if cell is None or cell.type == 'goal':
                        visited.add((nx, ny))
                        queue.append((nx, ny))
                        
        return False

class CustomUnlock8x8Env(MiniGridEnv):
    def __init__(self, **kwargs):
        mission_space = MissionSpace(mission_func=lambda: "open the door")
        super().__init__(
            mission_space=mission_space,
            grid_size=8,
            max_steps=500,
            **kwargs
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Draw a vertical wall down the middle (column 4)
        self.grid.vert_wall(4, 0, height)

        # Place a locked yellow door somewhere in that wall
        door_y = self._rand_int(1, height - 1)
        self.door = Door('yellow', is_locked=True)
        self.grid.set(4, door_y, self.door)

        # Place the key randomly on the left side of the wall
        self.place_obj(Key('yellow'), top=(1, 1), size=(3, 6))

        # Place the agent randomly on the left side
        self.place_agent(top=(1, 1), size=(3, 6))

        self.mission = "open the door"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.toggle:
            if self.door.is_open:
                reward = self._reward()
                terminated = True

        return obs, reward, terminated, truncated, info
class CompositeGridEnv(MiniGridEnv):
    def __init__(self, **kwargs):
        self.sub_envs = [
            CustomUnlock8x8Env(),
            CustomUnlock8x8Env(),
            CustomUnlock8x8Env(),
            CustomUnlock8x8Env()
        ]
        mission_space = MissionSpace(mission_func=lambda: "solve the active quadrant and reach the goal")
        super().__init__(
            grid_size=15, 
            max_steps=600, 
            mission_space=mission_space, 
            **kwargs
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        offsets = [(0, 0), (7, 0), (0, 7), (7, 7)]
        self.active_idx = random.randint(0, 3) # Save active index to instance

        for idx, (sub_env, (off_x, off_y)) in enumerate(zip(self.sub_envs, offsets)):
            sub_env.reset()
            for x in range(1, 8):
                for y in range(1, 8):
                    obj = sub_env.grid.get(x, y)
                    if obj is not None:
                        self.grid.set(off_x + x, off_y + y, obj)

            if idx == self.active_idx:
                self.agent_pos = (off_x + sub_env.agent_pos[0], off_y + sub_env.agent_pos[1])
                self.agent_dir = sub_env.agent_dir
                self.mission = sub_env.mission

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated or truncated:
            return obs, reward, terminated, truncated, info


        if action == self.actions.toggle:
            fwd_cell = self.grid.get(*self.front_pos)
            if fwd_cell is not None and fwd_cell.type == 'door' and fwd_cell.is_open:
                reward = self._reward()
                terminated = True

        return obs, reward, terminated, truncated, info

# Visualization Script
if __name__ == "__main__":
    env = CompositeGridEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    img = env.render()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Mission: {env.mission}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()