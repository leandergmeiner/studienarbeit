import gymnasium
import vizdoom as vzd
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def create_vizdoom_env():
    game = vzd.DoomGame()
    game.load_config("scenarios/shooting/defend_the_line.cfg")
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
            self.health = 100
            self.damage_taken = 0
            self.HITCOUNT = 0
            self.episode_count = 0
            self.timesteps_survived = 0
            self.writer = SummaryWriter(f"custom_metrics/env_{self.episode_count}")
        
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.game.new_episode()
            self.kill_State = 0
            state = self.game.get_state()
            return self._process_observation(state), {}

        #This is were all the reward shaping and custom metric collection happens
        def step(self, action):
            if self.game.is_episode_finished():
                return self._episode_done_response()

            # Directly checking for the test runs for how long agent stayed alive
            self.timesteps_survived += 1
            action = action.astype(int).tolist()
            reward = self.game.make_action(action)

            if self.game.get_state() is not None:
                game_vars = self.game.get_state().game_variables
                HITCOUNT, KILLCOUNT, DAMAGE_TAKEN, POS_X, POS_Y, AMMO = game_vars
                self._log_scalar("Game/PosX", POS_X)
                self._log_scalar("Game/PosY", POS_Y)

                living_reward = 0.01
                #Generally we just want him to live long and realize shooting enemies helps, while avoiding their shots increases his liftime
                reward += living_reward
                reward += (HITCOUNT - self.HITCOUNT) * 50 - DAMAGE_TAKEN * 0.1
                self.HITCOUNT = HITCOUNT

                if action[2]:  # move forward
                    reward -= 2
                # Since he had the full action space, for which this scenario was not intended, he kept moving forward. The idea of this penalty is that he should remain on his side of the map.
                # Value came by analyzing the map config file .wad
                if POS_X >= -465:
                    reward -= 10
                # Across all scenarios where shooting was involved, the agent had some kind of kink on shooting against walls. Up until now I can't explain why...but this penalizes him for doing so..
                if self._is_near_wall() and action[2]:
                    reward -= 100

            if self.game.is_episode_finished():
                return self._episode_done_response()

            state = self.game.get_state()
            if state is None:
                return np.zeros(self.observation_space.shape, dtype=np.uint8), reward, True, False, {}

            return self._process_observation(state), reward, False, False, {}

        def _episode_done_response(self):
            ammo = self.game.get_game_variable(vzd.SELECTED_WEAPON_AMMO)
            health = self.game.get_game_variable(vzd.HEALTH)
            outcome = 0 if health <= 0 else 1

            self._log_scalar("Game/Kills", self.HITCOUNT)
            self._log_scalar("Game/Outcome", outcome)
            self._log_scalar("Game/AmmoRemaining", ammo)
            self._log_scalar("Game/Health", health)
            self._log_scalar("Game/TimestepsSurvived", self.timesteps_survived)

            self.episode_count += 1
            self.kill_State = 0
            self.game.new_episode()

            return np.zeros(self.observation_space.shape, dtype=np.uint8), 0.0, True, False, {}

        def _log_scalar(self, tag, value):
            self.writer.add_scalar(tag, value, self.episode_count)

        def _is_near_wall(self):
            state = self.game.get_state()
            if state is None or state.depth_buffer is None:
                return False
            return np.min(state.depth_buffer) < 0  # WALL_THRESHOLD

        def _process_observation(self, state):
            if state is None:
                return np.zeros(self.observation_space.shape, dtype=np.uint8)
            return state.screen_buffer.copy()

        def render(self):
            state = self.game.get_state()
            return state.screen_buffer if state else np.zeros(self.observation_space.shape, dtype=np.uint8)

        def close(self):
            self.writer.close()
            self.game.close()

    return VizDoomGymEnv(game)
