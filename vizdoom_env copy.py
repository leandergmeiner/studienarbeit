import gymnasium
import vizdoom as vzd
import numpy as np
from haversine import haversine, Unit

def create_vizdoom_env():
    # Create the Doom environment
    game = vzd.DoomGame()
    
    # Set up Doom scenario
    game.load_config('defend_the_line.cfg')  
    # Set up Gym environment
    class VizDoomGymEnv(gymnasium.Env):
        def __init__(self, game):
            self.game = game
            self.game.init()
            self.step_count =0
            # Define the action and observation space
            self.action_space = gymnasium.spaces.MultiBinary(len(self.game.get_available_buttons())) # since Actions are passed in binary
            self.observation_space = gymnasium.spaces.Box(
                low=0, 
                high=255, 
                shape=(self.game.get_screen_height(), self.game.get_screen_width(), 3), 
                dtype=np.uint8
            )
            self.frame_skip = 4
            self.temp_state = []
            
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.game.new_episode()
            state = self.game.get_state()
            return (
                state.screen_buffer if state else np.zeros(self.observation_space.shape, dtype=np.uint8), 
                {}
            )
        
        
        def step(self, action):
            total_reward =0
            action = action.astype(int).tolist()
            reward = self.game.make_action(action, self.frame_skip) 
            try:
                vars = self.game.get_state().game_variables
                if len(self.temp_state)==0:
                    self.temp_state = vars[0:2]
                    self.last_pos = (vars[3], vars[4])
                    total_reward =reward
                else:
                    state = vars[0:2]
                    damage = vars[2]
                    position = (vars[3], vars[4])
                    r = self.reward_shaper(state, damage, position)
                    total_reward = reward +r
                
            except Exception as e: 
                total_reward = reward
            done = self.game.is_episode_finished()
            if done:
                self.game.new_episode()
            return self.game.get_state().screen_buffer, reward, done,False, {}

        def reward_shaper(self, state, damage, position):
            total_reward = 0
            for last,next in zip(self.temp_state, state):
                if last-next>=0: total_reward += 0.5
            
            if damage >=5: total_reward -= -0.1
            else: total_reward+=0.1

            # if haversine(self.last_pos, position, unit=Unit.METERS)!=0: total_reward+=0.01
            return total_reward
               
        def render(self):
            return self.game.get_state().screen_buffer
        
        def close(self):
            self.game.close()

    # Return the custom Gym environment
    return VizDoomGymEnv(game)
