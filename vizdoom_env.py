import gymnasium
import vizdoom as vzd
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def create_vizdoom_env():
    game = vzd.DoomGame()
    game.load_config("scenarios/move_n_avoid/deadly_corridor.cfg")
    game.set_depth_buffer_enabled(True) 
    game.set_labels_buffer_enabled(True)
    game.set_objects_info_enabled(True)
    
    class VizDoomGymEnv(gymnasium.Env):
        def __init__(self, game):
            self.game = game
            self.game.init()
            self.step_count = 0
            self.action_space = gymnasium.spaces.MultiBinary(len(self.game.get_available_buttons())) 
            self.observation_space = gymnasium.spaces.Box(
                low=0, 
                high=255, 
                shape=(self.game.get_screen_height(), self.game.get_screen_width(), 3),  
                dtype=np.uint8
            )
            self.kill_State = 0   
            self.ammo = 52
            self.health = 100
            self.damage_taken = 0
            self.HITCOUNT = 0
            self.episode_count = 0
            self.writer = SummaryWriter(f"custom_metrics/env_0")
            
        
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.game.new_episode()
            self.kill_State = 0
            self.ammo = 52
            self.health = 100
            self.damage_taken = 0
            self.HITCOUNT = 0
            state = self.game.get_state()
            return self._process_observation(state), {}

        # Applying reward shaping and logging custom metrics
        # We experimented a lot with reward shaping, generally we tracked everything we tried,but we can't push all of it. This marks the final reward shaping. 
        # We divide the final return by 1000, because we found some sources suggesting that this might improve model performance. 
        def step(self, action):
            done = self.game.is_episode_finished()
            action = action.astype(int).tolist()
            reward = self.game.make_action(action)
            
            if self.game.get_state() is not None:
                game_variables = self.game.get_state().game_variables
                health, damage_taken, HITCOUNT, KILLCOUNT, ammo, position_x = game_variables
                reward += position_x * 0.01moving
                health_delta = health - self.health
                self.health = health
                HITCOUNT_delta = HITCOUNT - self.HITCOUNT
                self.HITCOUNT = HITCOUNT
                ammo_delta = ammo - self.ammo
                self.ammo = ammo
                if HITCOUNT_delta > 0:
                    reward += HITCOUNT * 5
                if action[2] and HITCOUNT_delta <= 0:
                    reward -= 2.0
            
            if done:
                ammo = self.game.get_game_variable(vzd.SELECTED_WEAPON_AMMO)
                health = self.game.get_game_variable(vzd.HEALTH)
                position_x = self.game.get_game_variable(vzd.POSITION_X)
                self.game.new_episode()
                self.kill_State = 0
                outcome = 0 if health <= 0 else 1
                self.writer.add_scalar("Game/Kills", self.HITCOUNT, self.episode_count)
                self.writer.add_scalar("Game/Outcome", outcome, self.episode_count)
                self.writer.add_scalar("Game/AmmoRemaining", ammo, self.episode_count)
                self.writer.add_scalar("Game/Health", health, self.episode_count)
                self.writer.add_scalar("Game/Reward", reward, self.episode_count)
                self.writer.add_scalar("Game/Progress", position_x, self.episode_count)
                
                
                
                self.episode_count += 1
                return (
                    np.zeros(self.observation_space.shape, dtype=np.uint8), 
                    reward / 1000,  
                    done,
                    False,  
                    {},  
                )
            
            state = self.game.get_state()
            if state is None:
                return (
                    np.zeros(self.observation_space.shape, dtype=np.uint8), 
                    reward / 1000,
                    done,
                    False,
                    {},
                )
            
            return self._process_observation(state), reward / 1000, done, False, {}
        
        def _is_near_wall(self):
            state = self.game.get_state()
            if state is None:
                return False
            depth_buffer = state.depth_buffer
            if depth_buffer is None:
                return False
            
            min_depth = np.min(depth_buffer)
            WALL_THRESHOLD = 0  # Adjust as needed
            return min_depth < WALL_THRESHOLD

        # This function intially existed ,because we tried to do image processing on the frames and analyze if it enhanced performance. Currently this isn`'t doing something valuable. '
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
