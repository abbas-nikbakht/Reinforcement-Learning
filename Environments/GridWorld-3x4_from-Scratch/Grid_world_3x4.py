import pygame
import sys
import numpy as np


# Environment creation By gymnasium Library
def Enviroment(Action,state_inital):
    states = {
        's0_0': np.array([0,0]),'s0_1': np.array([0,1]),'s0_2': np.array([0,2]),'s0_3': np.array([0,3]),
        's1_0': np.array([1,0]),'s1_1': np.array([1,1]),'s1_2': np.array([1,2]),'s1_3': np.array([1,3]),
        's2_0': np.array([2,0]),'s2_1': np.array([2,1]),'s2_2': np.array([2,2]),'s2_3': np.array([2,3]),
    }
    
    Actions = {
        'Up': np.array([-1,0]),
        'Down': np.array([1,0]),
        'Right': np.array([0,1]),
        'Left': np.array([0,-1])
        } 
    
    
    # for Agent_next
    if Action=='Right':
        s_next= states[state_inital]+Actions['Right']
        for k, v in states.items():
            if (v == s_next).all():
                s_next=k 
                
        s_next_Up= states[state_inital]+Actions['Up'] # for Deviation to the sides
        for k, v in states.items():
            if (v == s_next_Up).all():
                s_next_Up=k 
                
        s_next_Down= states[state_inital]+Actions['Down'] # for Deviation to the sides
        for k, v in states.items():
            if (v == s_next_Down).all():
                s_next_Down=k 
            
            
        #Obstacles Up environment
        if (s_next_Up == np.array([-1,0])).all():
            s_next_Up='s0_0'   
        if (s_next_Up == np.array([-1,1])).all():
            s_next_Up='s0_1'
        if (s_next_Up == np.array([-1,2])).all():
            s_next_Up='s0_2'  
        if (s_next_Up == np.array([-1,3])).all():
            s_next_Up='s0_3'        
     
        #Obstacles Right environment
        if (s_next == np.array([0,4])).all():
            s_next='s0_3'
        if (s_next == np.array([1,4])).all():
            s_next='s1_3'
        if (s_next == np.array([2,4])).all():
            s_next='s2_3'
            
        #Obstacles Down environment
        if (s_next_Down == np.array([3,3])).all():
            s_next_Down='s2_3'
        if (s_next_Down == np.array([3,2])).all():
            s_next_Down='s2_2'
        if (s_next_Down == np.array([3,1])).all():
            s_next_Down='s2_1'
        if (s_next_Down == np.array([3,0])).all():
            s_next_Down='s2_0'
                        
            
        #Obstacles center environment
        if s_next== 's1_1':
            s_next= state_inital
        if s_next_Down== 's1_1':
            s_next_Down= state_inital
        if s_next_Up== 's1_1':
            s_next_Up= state_inital
                        
            
        
        p_model= {
            s_next: 0.8,
            s_next_Up: 0.1,
            s_next_Down: 0.1
            }
        s_next_all=  {
            s_next: s_next,
            s_next_Up: s_next_Up,
            s_next_Down: s_next_Down
            }

        r=  {
            s_next: -0.04,
            s_next_Up: -0.04,
            s_next_Down: -0.04,
            }
        # Reward terminal
        for i in r:
            if i== 's0_3':
                r[i]= +1-0.04
            if i== 's1_3':
                r[i]= -1-0.04
        

        Agent_next= np.random.choice([s_next,s_next_Up,s_next_Down],p=[0.8,0.1,0.1])
        Agent_next= str(Agent_next) 

    elif Action=='Left':
    
        s_next= states[state_inital]+Actions['Left']
        for k, v in states.items():
            if (v == s_next).all():
                s_next=k 
                
        s_next_Up= states[state_inital]+Actions['Up'] # for Deviation to the sides
        for k, v in states.items():
            if (v == s_next_Up).all():
                s_next_Up=k 
                
        s_next_Down= states[state_inital]+Actions['Down'] # for Deviation to the sides
        for k, v in states.items():
            if (v == s_next_Down).all():
                s_next_Down=k 

            
        #Obstacles Up environment
        if (s_next_Up == np.array([-1,0])).all():
            s_next_Up='s0_0'   
        if (s_next_Up == np.array([-1,1])).all():
            s_next_Up='s0_1'
        if (s_next_Up == np.array([-1,2])).all():
            s_next_Up='s0_2'  
        if (s_next_Up == np.array([-1,3])).all():
            s_next_Up='s0_3'          
            
        #Obstacles Down environment
        if (s_next_Down == np.array([3,3])).all():
            s_next_Down='s2_3'
        if (s_next_Down == np.array([3,2])).all():
            s_next_Down='s2_2'
        if (s_next_Down == np.array([3,1])).all():
            s_next_Down='s2_1'
        if (s_next_Down == np.array([3,0])).all():
            s_next_Down='s2_0'
            
        #Obstacles Left environment
        if (s_next == np.array([2,-1])).all():
            s_next='s2_0'
        if (s_next == np.array([1,-1])).all():
            s_next='s1_0'
        if (s_next == np.array([0,-1])).all():
            s_next='s0_0'   

        #Obstacles center environment
        if s_next_Up== 's1_1':
            s_next_Up= state_inital
        if s_next_Down== 's1_1':
            s_next_Down= state_inital
        if s_next== 's1_1':
            s_next= state_inital


        p_model= {
            s_next: 0.8,
            s_next_Up: 0.1,
            s_next_Down: 0.1
            }
        s_next_all=  {
            s_next: s_next,
            s_next_Up: s_next_Up,
            s_next_Down: s_next_Down
            }


        r=  {
            s_next: -0.04,
            s_next_Up: -0.04,
            s_next_Down: -0.04,
            }                 
        # Reward terminal
        for i in r:
            if i== 's0_3':
                r[i]= +1-0.04
            if i== 's1_3':
                r[i]= -1-0.04
                
        Agent_next= np.random.choice([s_next,s_next_Up,s_next_Down],p=[0.8,0.1,0.1])
        Agent_next= str(Agent_next) 

        
    elif Action=='Up':
    
        s_next= states[state_inital]+Actions['Up']
        for k, v in states.items():
            if (v == s_next).all():
                s_next=k 
                

        s_next_Right= states[state_inital]+Actions['Right'] # for Deviation to the sides
        for k, v in states.items():
            if (v == s_next_Right).all():
                s_next_Right=k 
                
                
        s_next_Left= states[state_inital]+Actions['Left'] # for Deviation to the sides
        for k, v in states.items():
            if (v == s_next_Left).all():
                s_next_Left=k 


        #Obstacles Up environment
        if (s_next == np.array([-1,0])).all():
            s_next='s0_0'   
        if (s_next == np.array([-1,1])).all():
            s_next='s0_1'
        if (s_next == np.array([-1,2])).all():
            s_next='s0_2'  
        if (s_next == np.array([-1,3])).all():
            s_next='s0_3'        
     
        #Obstacles Right environment
        if (s_next_Right == np.array([0,4])).all():
            s_next_Right='s0_3'
        if (s_next_Right == np.array([1,4])).all():
            s_next_Right='s1_3'
        if (s_next_Right == np.array([2,4])).all():
            s_next_Right='s2_3'

            
        #Obstacles Left environment
        if (s_next_Left == np.array([2,-1])).all():
            s_next_Left='s2_0'
        if (s_next_Left == np.array([1,-1])).all():
            s_next_Left='s1_0'
        if (s_next_Left == np.array([0,-1])).all():
            s_next_Left='s0_0'               
           
        #Obstacles center environment
        if s_next== 's1_1':
            s_next= state_inital
        if s_next_Right== 's1_1':
            s_next_Right= state_inital
        if s_next_Left== 's1_1':
            s_next_Left= state_inital
         
               
        p_model= {
            s_next: 0.8,
            s_next_Right: 0.1,
            s_next_Left: 0.1
            }
        s_next_all=  {
            s_next: s_next,
            s_next_Right: s_next_Right,
            s_next_Left: s_next_Left
            }


        r=  {
            s_next: -0.04,
            s_next_Right: -0.04,
            s_next_Left: -0.04,
            }
        # Reward terminal
        for i in r:
            if i== 's0_3':
                r[i]= +1-0.04
            if i== 's1_3':
                r[i]= -1-0.04
        
        
        Agent_next= np.random.choice([s_next,s_next_Right,s_next_Left],p=[0.8,0.1,0.1])
        Agent_next= str(Agent_next) 

    elif Action=='Down':
    
        s_next= states[state_inital]+Actions['Down']
        for k, v in states.items():
            if (v == s_next).all():
                s_next=k 
                
        s_next_Left= states[state_inital]+Actions['Left'] # for Deviation to the sides
        for k, v in states.items():
            if (v == s_next_Left).all():
                s_next_Left=k 
                
        s_next_Right= states[state_inital]+Actions['Right'] # for Deviation to the sides
        for k, v in states.items():
            if (v == s_next_Right).all():
                s_next_Right=k 
     
        #Obstacles Right environment
        if (s_next_Right == np.array([0,4])).all():
            s_next_Right='s0_3'
        if (s_next_Right == np.array([1,4])).all():
            s_next_Right='s1_3'
        if (s_next_Right == np.array([2,4])).all():
            s_next_Right='s2_3'
            
        #Obstacles Down environment
        if (s_next == np.array([3,3])).all():
            s_next='s2_3'
        if (s_next == np.array([3,2])).all():
            s_next='s2_2'
        if (s_next == np.array([3,1])).all():
            s_next='s2_1'
        if (s_next == np.array([3,0])).all():
            s_next='s2_0'
            
        #Obstacles Left environment
        if (s_next_Left == np.array([2,-1])).all():
            s_next_Left='s2_0'
        if (s_next_Left == np.array([1,-1])).all():
            s_next_Left='s1_0'
        if (s_next_Left == np.array([0,-1])).all():
            s_next_Left='s0_0'               


        #Obstacles center environment
        if s_next== 's1_1':
            s_next= state_inital
        if s_next_Right== 's1_1':
            s_next_Right= state_inital
        if s_next_Left== 's1_1':
            s_next_Left= state_inital
            
        p_model= {
            s_next: 0.8,
            s_next_Right: 0.1,
            s_next_Left: 0.1
            }
        s_next_all=  {
            s_next: s_next,
            s_next_Right: s_next_Right,
            s_next_Left: s_next_Left
            }


        r=  {
            s_next: -0.04,
            s_next_Right: -0.04,
            s_next_Left: -0.04,
            }
        # Reward terminal
        for i in r:
            if i== 's0_3':
                r[i]= +1-0.04
            if i== 's1_3':
                r[i]= -1-0.04
        
                    
        Agent_next= np.random.choice([s_next,s_next_Left,s_next_Right],p=[0.8,0.1,0.1])
        Agent_next= str(Agent_next)
    
    
    

     # for corner  
    if len(s_next_all)==2:
        for i in s_next_all:
            if i=='s2_3':
                p_model['s2_3']= 0.9
                
            if i=='s0_0':
                p_model['s0_0']= 0.9
            
            if i=='s2_0':
                p_model['s2_0']= 0.9

    
    
    return r,s_next_all,p_model,Agent_next


# Function to display the environment graphically and play the game with a human agent.    
def Show_Game_Gridworld():
    
    pygame.init()  
    game_disply = pygame.display.set_mode((465,305))
    pygame.display.set_caption("My First")
    
    # Agent
    states = {
        's0_0': (60, 55),
        's0_1': (175, 55),
        's0_2': (290, 55),
        's0_3': (405, 55),
        
        's1_0': (60, 153),
        's1_1': (175, 153),
        's1_2': (290, 153),
        's1_3': (405, 153),
        
        's2_0': (60, 251),
        's2_1': (175, 251),
        's2_2': (290, 251),
        's2_3': (405, 251)
    }
    
    
    def Human(Human_in):
        pygame.draw.circle(game_disply, (0,0,0), Human_in, 15)
    
    
    #### Parameters
    human_init='s0_0'
    
    Terminal_states = ('s0_3', 's1_3')
    Human_next= human_init
    
    
    while True:
        
    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            if event.type == pygame.KEYDOWN and Human_next not in Terminal_states:
                
                if event.key == pygame.K_UP:
                    [r,s,p,Human_next]=Enviroment('Up',human_init)
                    print("Up")
                    print(r[Human_next])
                    
                if event.key == pygame.K_DOWN:
                    [r,s,p,Human_next]=Enviroment('Down',human_init)
                    print("DOWN")
                    print(r[Human_next])
    
                if event.key == pygame.K_LEFT:
                    [r,s,p,Human_next]=Enviroment('Left',human_init)
                    print("LEFT")
                    print(r[Human_next])
    
                if event.key == pygame.K_RIGHT:
                    [r,s,p,Human_next]=Enviroment('Right',human_init)
                    print("RIGHT")
                    print(r[Human_next])
    
                      
                          
                if Human_next=='s0_3':
        
                    print('You win, Agent')
        
                if Human_next=='s1_3':
        
                    print('You lost, Agent')            
    
    
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
    
    
        
        Human(states[Human_next])                         
        human_init=Human_next                
                    
        pygame.display.update()


# Display the gridworld environment and then we can move towards the target,
# which is the green house, using the keyboard directions.
Show_Game_Gridworld()




