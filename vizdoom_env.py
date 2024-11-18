import gymnasium
import vizdoom as vzd
import numpy as np

def create_vizdoom_env():
    # Create the Doom environment
    game = vzd.DoomGame()
    
    # Set up Doom scenario
    game.load_config('basic.cfg')  

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
            
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.game.new_episode()
            state = self.game.get_state()
            return (
                state.screen_buffer if state else np.zeros(self.observation_space.shape, dtype=np.uint8), 
                {}
            )
        
        
        def step(self, action):
            action = action.astype(int).tolist()
            reward = self.game.make_action(action)  
            done = self.game.is_episode_finished()
            self.step_count += 1  
            
            # Truncated => Conditional episode_finish
            if self.step_count >= 20:
                truncated = True
            else:
                truncated = False

            if done:
                self.game.new_episode()
            return self.game.get_state().screen_buffer, reward, done,truncated, {}

        def render(self):
            return self.game.get_state().screen_buffer
        
        def close(self):
            self.game.close()

    # Return the custom Gym environment
    return VizDoomGymEnv(game)
