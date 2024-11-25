import gymnasium
import vizdoom as vzd
import numpy as np
from haversine import haversine, Unit
from torch.utils.tensorboard import SummaryWriter
def create_vizdoom_env():
    game = vzd.DoomGame()
    game.load_config("scenarios/move_n_avoid/deadly_corridor.cfg")
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
            self.kill_State = []        
            self.game_sucess = +200
            self.max_ammo = 52
            self.writer = SummaryWriter("./custom_metrics")
            self.episode_count = 0
            
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.game.new_episode()
            self.kill_State = []
            state = self.game.get_state()
            return (
                np.expand_dims(state.screen_buffer, axis=-1)  if state else np.zeros(self.observation_space.shape, dtype=np.uint8), 
                {}
            )
        
        
        def step(self, action):
            total_reward =0
            done = self.game.is_episode_finished()
            action = action.astype(int).tolist()
            reward = self.game.make_action(action)/100
            #Penalty for shooting
            if action[2]: reward -= 1

            # Lass icj jetzt auch erstmal außen vor, da nicht primary goal.
            # if self.game.get_game_variable(vzd.SELECTED_WEAPON_AMMO)==0: done =True
            
            health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
            kill_count = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
            if done:
                ammo = self.game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
                self.game.new_episode()
                self.kill_State=[]
  
                if health <=0: 
                    outcome = "Death"
                else: 
                    outcome = "Win"
                
                self.writer.add_scalar("Game/Kills", kill_count, self.episode_count)
                self.writer.add_scalar("Game/Outcome", {"Death": 0, "Win": 1}[outcome], self.episode_count)
                self.writer.add_scalar("Game/AmmoRemaining", ammo, self.episode_count)


                self.episode_count +=1
                # if kill_count!=0 and kill_count/self.max_ammo - self.game.get_game_variable(vzd.SELECTED_WEAPON_AMMO)>=0.5 : total_reward+=0.1
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
                if len(self.kill_State)==0:
                    self.kill_State = [vars[0]]
                    total_reward = reward
                else:
                    damage = vars[1]
                    r = self.reward_shaper(kill_count, damage)
                    total_reward = reward +r   
                    self.kill_State = [vars[0]]             
            except Exception as e: 
                print(f"Error during step: {e}")
            return np.expand_dims(self.game.get_state().screen_buffer, axis=-1) , total_reward, done,False, {}
                    
        def reward_shaper(self, kill_count , damage):
            total_reward = 0
            if self.kill_State[0]-kill_count>0: total_reward += 5            
            if damage >=1: total_reward -= damage *2
            # depth_buffer = game.get_state().depth_buffer
            # center_x = len(depth_buffer[0])//2
            # center_y = len(depth_buffer) //2
            # front_distance = depth_buffer[center_x][center_y]
            # if front_distance<=5: total_reward-=0.1
            return total_reward
               
        def render(self):
            return self.game.get_state().screen_buffer
        
        def close(self):
            self.writer.close() 
            self.game.close()

    return VizDoomGymEnv(game)
