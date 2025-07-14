import gymnasium
import vizdoom as vzd
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def create_vizdoom_env():
    game = vzd.DoomGame()
    game.load_config("scenarios/move_n_avoid/take_cover.cfg")
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_objects_info_enabled(True)
    
    class VizDoomGymEnv(gymnasium.Env):
        def __init__(self, game):
            self.game = game
            self.game.init()
            self.step_count = 0
            self.available_actions = [
            [0, 0],  # no action
            [1, 0],  # move left
            [0, 1]]  # move right
            self.action_space = gymnasium.spaces.Discrete(len(self.available_actions))
            self.observation_space = gymnasium.spaces.Box(
                low=0,
                high=255,
                shape=(self.game.get_screen_height(), self.game.get_screen_width(), 3),
                dtype=np.uint8
            )
            # For repeated-action penalty
            self.last_action = None
            self.repeat_count = 0
            self.repeat_threshold = 5  # number of repeats allowed without penalty
            self.repeat_penalty = 0.5 # penalty per extra repeat
            
            self.kill_State = 0
            self.health = 100
            self.episode_count = 0
            self.writer = SummaryWriter(f"custom_metrics/env_{self.episode_count}")
        
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.game.new_episode()
            self.kill_State = 0
            self.health = 100
            self.step_count = 0
            self.last_action = None
            self.repeat_count = 0
            self.writer = SummaryWriter(f"custom_metrics/env_{self.episode_count}")
            state = self.game.get_state()
            return self._process_observation(state), {}
        
        def step(self, action):
            self.step_count += 1
            done = self.game.is_episode_finished()
            action_list = self.available_actions[action]
            reward = self.game.make_action(action_list)

            # Repeated action penalty
            #if self.last_action is not None and np.array_equal(action, self.last_action):
                #self.repeat_count += 1
            #else:
                #self.repeat_count = 0
            #if self.repeat_count > self.repeat_threshold:
                #penalty_amount = self.repeat_penalty * (self.repeat_count - self.repeat_threshold)
                #reward -= penalty_amount
                #self.writer.add_scalar("Game/RepeatActionPenalty", -penalty_amount, self.step_count)
            #self.last_action = action.copy()

            if self.game.get_state() is not None:
                state = self.game.get_state()
                game_variables = state.game_variables
                health, position_y, position_x = game_variables
                health_delta = health - self.health
                if health_delta < 0:
                    reward -= 0.5
                    self.writer.add_scalar("Game/HealthDelta", health_delta, self.step_count)
                self.health = health
                self.writer.add_scalar("Game/Health", health, self.step_count)

                #Change wall logic, to the y positions of the wall and penalize if he's wallhugging
                if self.best_vision(position_y):
                    reward+= 1                
                else:
                    reward -= 2
                self.writer.add_scalar("Game/Position", position_y, self.step_count)

            if done:
                self.writer.close()
                return (np.zeros(self.observation_space.shape, dtype=np.uint8), reward, done, False, {})

            state = self.game.get_state()
            if state is None:
                return (np.zeros(self.observation_space.shape, dtype=np.uint8), reward, done, False, {})
        
            return self._process_observation(state), reward, done, False, {}

        def best_vision(self, y):
            """Check if agent ran into a wall, threshold 50"""
            return y <= 484 or y >= 284 

        def _process_observation(self, state):
            if state is None:
                return np.zeros(self.observation_space.shape, dtype=np.uint8)
            screen_buffer = state.screen_buffer.copy()
            return screen_buffer

        def render(self):
            state = self.game.get_state()
            if state is None:
                return np.zeros(self.observation_space.shape, dtype=np.uint8)
            return state.screen_buffer

        def close(self):
            self.writer.close()
            self.game.close()

    return VizDoomGymEnv(game)
