import gymnasium
import vizdoom as vzd
import numpy as np
from haversine import haversine, Unit

def create_vizdoom_env():
    game = vzd.DoomGame()
    game.load_config("scenarios/shooting/basic.cfg")
    # game.add_game_args("+sv_infiniteammo 1")
    game.set_depth_buffer_enabled(True)
    # Set up Gym environment
    class VizDoomGymEnv(gymnasium.Env):
        def __init__(self, game):
            self.game = game
            self.game.init()
            self.step_count =0
            self.action_space = gymnasium.spaces.MultiBinary(len(self.game.get_available_buttons())) # since Actions are passed in binary
            self.observation_space = gymnasium.spaces.Box(
                low=0, 
                high=255, 
                shape=(self.game.get_screen_height(), self.game.get_screen_width(),1), 
                dtype=np.uint8
            )
            self.frame_skip = 4
            self.temp_state = []
            self.game_sucess = +200
            self.max_ammo = 50
            
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.game.new_episode()
            self.temp_state = []
            state = self.game.get_state()
            return (
                np.expand_dims(state.screen_buffer, axis=-1)  if state else np.zeros(self.observation_space.shape, dtype=np.uint8), 
                {}
            )
        
        
        def step(self, action):
            total_reward =0
            done = self.game.is_episode_finished()
            action = action.astype(int).tolist()
            # reward = self.game.make_action(action, self.frame_skip) 
            reward = self.game.make_action(action) 
            if self.game.get_game_variable(vzd.SELECTED_WEAPON_AMMO)==0: done =True
            health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
            kill_count = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
            if done:
                self.game.new_episode()
                #Change it from rewarding to penalizing
                if health ==0 or kill_count==0: 
                    total_reward = reward - self.game_sucess
                # else:  total_reward = reward - self.game_ended
                if kill_count!=0 and self.game.get_game_variable(vzd.SELECTED_WEAPON_AMMO)/self.max_ammo>=0.95: total_reward+=0.5
                return (
                    np.zeros(self.observation_space.shape, dtype=np.uint8), 
                    total_reward,  
                    done,
                    False,  
                    {},  
                )

            state = self.game.get_state()
            if state is None:
                return (
                    np.zeros(self.observation_space.shape, dtype=np.uint8), 
                    total_reward,
                    done,
                    False,
                    {},
                )

            try:
                vars = state.game_variables
                if len(self.temp_state)==0:
                    self.temp_state = vars[0:2]
                    # self.last_pos = (vars[3], vars[4])
                    total_reward =reward
                else:
                    stats = vars[0:2]
                    damage = vars[2]
                    # position = (vars[3], vars[4])
                    r = self.reward_shaper(stats, damage)
                    total_reward = reward +r                
            except Exception as e: 
                print(f"Error during step: {e}")
            return np.expand_dims(self.game.get_state().screen_buffer, axis=-1) , total_reward, done,False, {}
                    
        def reward_shaper(self, state, damage):
            total_reward = 0
            for last,next in zip(self.temp_state, state):
                if last-next>0: total_reward += 100
            
            if damage >=5: total_reward -= -0.5

            # Calculate distance moved using Euclidean distance
            # dx = self.last_pos[0] - position[0]
            # dy = self.last_pos[1] - position[1]
            # distance_moved = np.sqrt(dx**2 + dy**2)
            # if distance_moved>50: total_reward+=0.001

            depth_buffer = game.get_state().depth_buffer
            center_x = len(depth_buffer[0])//2
            center_y = len(depth_buffer) //2
            front_distance = depth_buffer[center_x][center_y]
            if front_distance<=5: total_reward-=0.05
            return total_reward
               
        def render(self):
            return self.game.get_state().screen_buffer
        
        def close(self):
            self.game.close()

    # Return the custom Gym environment
    return VizDoomGymEnv(game)
