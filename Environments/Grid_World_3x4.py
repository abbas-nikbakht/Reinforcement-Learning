import pygame
import sys
import gymnasium as gym

# Environment creation.
class GridWorld3x4():
    
    def __init__(self):
        ##
        self.env = gym.make(
            "FrozenLake-v1",
            desc=["SFFG",
                  "FFFH",
                  "FFFF"],
            is_slippery=True,
        )
        
        
        # state 0 
        self.env.unwrapped.P[0][0] = [
            (0.8, 0, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 0, -0.04, False),
            (0.1, 4, -0.04, False)
        ]
        
        self.env.unwrapped.P[0][1] = [
            (0.8, 4, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 0, -0.04, False),
            (0.1, 1, -0.04, False)
        ]
        
        self.env.unwrapped.P[0][2] = [
            (0.8, 1, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 0, -0.04, False),
            (0.1, 4, -0.04, False)
        ]
        
        self.env.unwrapped.P[0][3] = [
            (0.8, 0, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 0, -0.04, False),
            (0.1, 1, -0.04, False)
        ]
        
        # state 1 
        self.env.unwrapped.P[1][0] = [
            (0.8, 0, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 1, -0.04, False),
            (0.1, 1, -0.04, False)
        ]
        self.env.unwrapped.P[1][1] = [
            (0.8, 1, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 0, -0.04, False),
            (0.1, 2, -0.04, False)
        ]
        self.env.unwrapped.P[1][2] = [
            (0.8, 2, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 1, -0.04, False),
            (0.1, 1, -0.04, False)
        ]
        self.env.unwrapped.P[1][3] = [
            (0.8, 1, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 0, -0.04, False),
            (0.1, 2, -0.04, False)
        ]
        
        # state 2 
        self.env.unwrapped.P[2][0] = [
            (0.8, 1, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 2, -0.04, False),
            (0.1, 6, -0.04, False)
        ]
        self.env.unwrapped.P[2][1] = [
            (0.8, 6, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 1, -0.04, False),
            (0.1, 3, 1-0.04, True)
        ]
        self.env.unwrapped.P[2][2] = [
            (0.8, 3, 1-0.04, True), # (prob, next_state, reward, terminated)
            (0.1, 2, -0.04, False),
            (0.1, 6, -0.04, False)
        ]
        self.env.unwrapped.P[2][3] = [
            (0.8, 2, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 3, 1-0.04, True),
            (0.1, 1, -0.04, False)
        ]
        
        # state 3 
         # Terminal
         
        # state 4 
        self.env.unwrapped.P[4][0] = [
            (0.8, 4, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 0, -0.04, False),
            (0.1, 8, -0.04, False)
        ]
        self.env.unwrapped.P[4][1] = [
            (0.8, 8, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 4, -0.04, False),
            (0.1, 4, -0.04, False)
        ]
        self.env.unwrapped.P[4][2] = [
            (0.8, 4, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 0, -0.04, False),
            (0.1, 8, -0.04, False)
        ]
        self.env.unwrapped.P[4][3] = [
            (0.8, 0, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 4, -0.04, False),
            (0.1, 4, -0.04, False)
        ]
        
        # state 5 
            # Wall
            
        # state 6 
        self.env.unwrapped.P[6][0] = [
            (0.8, 6, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 2, -0.04, False),
            (0.1, 10, -0.04, False)
        ]
        self.env.unwrapped.P[6][1] = [
            (0.8, 10, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 6, -0.04, False),
            (0.1, 7, -1-0.04, True)
        ]
        self.env.unwrapped.P[6][2] = [
            (0.8, 7, -1-0.04, True), # (prob, next_state, reward, terminated)
            (0.1, 2, -0.04, False),
            (0.1, 10, -0.04, False)
        ]
        self.env.unwrapped.P[6][3] = [
            (0.8, 2, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 6, -0.04, False),
            (0.1, 7, -1-0.04, True)
        ]
        
        # state 7 
            # Terminal
            
        # state 8 
        self.env.unwrapped.P[8][0] = [
            (0.8, 8, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 4, -0.04, False),
            (0.1, 8, -0.04, False)
        ]
        self.env.unwrapped.P[8][1] = [
            (0.8, 8, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 8, -0.04, False),
            (0.1, 9, -0.04, False)
        ]
        self.env.unwrapped.P[8][2] = [
            (0.8, 9, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 4, -0.04, False),
            (0.1, 8, -0.04, False)
        ]
        self.env.unwrapped.P[8][3] = [
            (0.8, 4, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 9, -0.04, False),
            (0.1, 8, -0.04, False)
        ]
        
        # state 9 
        self.env.unwrapped.P[9][0] = [
            (0.8, 8, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 9, -0.04, False),
            (0.1, 9, -0.04, False)
        ]
        self.env.unwrapped.P[9][1] = [
            (0.8, 9, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 8, -0.04, False),
            (0.1, 10, -0.04, False)
        ]
        self.env.unwrapped.P[9][2] = [
            (0.8, 10, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 9, -0.04, False),
            (0.1, 9, -0.04, False)
        ]
        self.env.unwrapped.P[9][3] = [
            (0.8, 9, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 8, -0.04, False),
            (0.1, 10, -0.04, False)
        ]
        
        # state 10 
        self.env.unwrapped.P[10][0] = [
            (0.8, 9, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 6, -0.04, False),
            (0.1, 10, -0.04, False)
        ]
        self.env.unwrapped.P[10][1] = [
            (0.8, 10, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 9, -0.04, False),
            (0.1, 11, -0.04, False)
        ]
        self.env.unwrapped.P[10][2] = [
            (0.8, 11, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 6, -0.04, False),
            (0.1, 10, -0.04, False)
        ]
        self.env.unwrapped.P[10][3] = [
            (0.8, 6, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 11, -0.04, False),
            (0.1, 9, -0.04, False)
        ]
        
        # state 11 
        self.env.unwrapped.P[11][0] = [
            (0.8, 10, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 7, -1-0.04, True),
            (0.1, 11, -0.04, False)
        ]
        self.env.unwrapped.P[11][1] = [
            (0.8, 11, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 10, -0.04, False),
            (0.1, 11, -0.04, False)
        ]
        self.env.unwrapped.P[11][2] = [
            (0.8, 11, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 7, -1-0.04, True),
            (0.1, 11, -0.04, False)
        ]
        self.env.unwrapped.P[11][3] = [
            (0.8, 7, -1-0.04, True), # (prob, next_state, reward, terminated)
            (0.1, 10, -0.04, False),
            (0.1, 11, -0.04, False)
        ]
        
        
        
    def step(self, state_initial, action): 
        ### state_inital
        self.env.reset()
        
        base = self.env.unwrapped
        base.s = state_initial 
        
        state, reward, terminated, truncated, info = self.env.step(action)

        return state, reward, terminated, truncated, info

    def reset(self,seed):
        self.env.reset(seed=seed)
 

# Function to display the environment graphically and play the game with a human agent.    
def Show_Game_Gridworld(name_Envi):
    
    pygame.init()      
    game_disply = pygame.display.set_mode((465,415))
    
    pygame.display.set_caption("My First")
    

    # Agent
    states = {
        0: (60, 55),
        1: (175, 55),
        2: (290, 55),
        3: (405, 55),
        
        4: (60, 153),
        5: (175, 153),
        6: (290, 153),
        7: (405, 153),
        
        8: (60, 251),
        9: (175, 251),
        10: (290, 251),
        11: (405, 251)
    }
    
    # Display initial
    #Row 4  print (parameters state,action,reward)
    pygame.draw.rect(game_disply, "black",(5,315,455,95))
    
    font = pygame.font.Font(None, 30)
    text = font.render(
        "State: 0       Action: -      Reward: 0 ",
        True,
        "white"
    )
    game_disply.blit(text, (20, 315))

    ## print Episode, Step
    font = pygame.font.Font(None, 30)
    text = font.render(
        "             Episode:              Step: 0",
        True,
        "white"
    )
    game_disply.blit(text, (20, 340))    
    
    ###
    def Human(Human_in):
        pygame.draw.circle(game_disply, (0,0,0), Human_in, 15)
    
    
    #### Parameters
    
    human_init=0
    
    terminated = False
    
    Human_next= human_init
    step=0
    
    while True:

        
        ## Visual appearance of the game environment
        #Row 1
        pygame.draw.rect(game_disply,(255,255,255),(5,5,110,95))
        pygame.draw.rect(game_disply,(255,255,255),(120,5,110,95))
        pygame.draw.rect(game_disply,(255,255,255),(235,5,110,95))
        pygame.draw.rect(game_disply,(0,255,0),(350,5,110,95))
        #Row 1
        pygame.draw.rect(game_disply,(255,255,255),(5,105,110,95))
        pygame.draw.rect(game_disply,(128,128,128),(120,105,110,95))
        pygame.draw.rect(game_disply,(255,255,255),(235,105,110,95))
        pygame.draw.rect(game_disply,(255,0,0),(350,105,110,95))
        #Row 1
        pygame.draw.rect(game_disply,(255,255,255),(5,205,110,95))
        pygame.draw.rect(game_disply,(255,255,255),(120,205,110,95))
        pygame.draw.rect(game_disply,(255,255,255),(235,205,110,95))
        pygame.draw.rect(game_disply,(255,255,255),(350,205,110,95))
          
    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            if event.type == pygame.KEYDOWN and terminated==False:
                    
                if event.key == pygame.K_UP:
                    Human_next, reward, terminated, truncated, info=name_Envi.step(human_init, 3)
                    print("Up")
                    print(reward)
                    step=step+1
                    #Row 4  print (parameters state,action,reward)
                    pygame.draw.rect(game_disply, "black",(5,315,455,95))
                    
                    font = pygame.font.Font(None, 30)
                    text = font.render(
                        f"State: {Human_next}       Action: Up      Reward: {reward} ",
                        True,
                        "white"
                    )
                    game_disply.blit(text, (20, 315))

                    ## print Episode, Step
                    font = pygame.font.Font(None, 30)
                    text = font.render(
                        f"             Episode:              Step: {step}",
                        True,
                        "white"
                    )
                    game_disply.blit(text, (20, 340))
                    
                if event.key == pygame.K_DOWN:
                    Human_next, reward, terminated, truncated, info=name_Envi.step(human_init, 1)
                    print("DOWN")
                    print(reward)
                    step=step+1
                    #Row 4 (parameters state,action,reward)
                    pygame.draw.rect(game_disply, "black",(5,315,455,95))
                    
                    font = pygame.font.Font(None, 30)
                    text = font.render(
                        f"State: {Human_next}       Action: DOWN      Reward: {reward}",
                        True,
                        "white"
                    )
                    game_disply.blit(text, (20, 315))
                    ## print Episode, Step
                    font = pygame.font.Font(None, 30)
                    text = font.render(
                        f"             Episode:              Step: {step}",
                        True,
                        "white"
                    )
                    game_disply.blit(text, (20, 340))
                    
                if event.key == pygame.K_LEFT:
                    Human_next, reward, terminated, truncated, info=name_Envi.step(human_init, 0)
                    print("LEFT")
                    print(reward)
                    step=step+1
                    #Row 4 (parameters state,action,reward)
                    pygame.draw.rect(game_disply, "black",(5,315,455,95))
                    
                    font = pygame.font.Font(None, 30)
                    text = font.render(
                        f"State: {Human_next}       Action: LEFT      Reward: {reward}",
                        True,
                        "white"
                    )
                    game_disply.blit(text, (20, 315))
                    ## print Episode, Step
                    font = pygame.font.Font(None, 30)
                    text = font.render(
                        f"             Episode:              Step: {step}",
                        True,
                        "white"
                    )
                    game_disply.blit(text, (20, 340))
                                       
                if event.key == pygame.K_RIGHT:
                    Human_next, reward, terminated, truncated, info=name_Envi.step(human_init, 2)
                    print("RIGHT")
                    print(reward)
                    step=step+1
                    #Row 4 (parameters state,action,reward)
                    pygame.draw.rect(game_disply, "black",(5,315,455,95))
                    
                    font = pygame.font.Font(None, 30)
                    text = font.render(
                        f"State: {Human_next}       Action: RIGHT      Reward: {reward}",
                        True,
                        "white"
                    )
                    game_disply.blit(text, (20, 315))
                    ## print Episode, Step
                    font = pygame.font.Font(None, 30)
                    text = font.render(
                        f"             Episode:              Step: {step}",
                        True,
                        "white"
                    )
                    game_disply.blit(text, (20, 340))
                       
                                     
                if Human_next==3:
        
                    print('Goal')
        
                if Human_next==7:
        
                    print('Hollow')            
    

        
    
        
        Human(states[Human_next])                         
        human_init=Human_next                
                    
        pygame.display.update()

                 
EnviGridWorld3x4 = GridWorld3x4()

# Display the gridworld environment and then we can move towards the target,
# which is the green house, using the keyboard directions.
Show_Game_Gridworld(EnviGridWorld3x4)



















