from dash import Dash, dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
from dash import ctx,no_update

######################  Description #######################

#        Grilled Word 3x4 Dynamic Programing example

############################################################

import gymnasium as gym
import random as random
import numpy as np
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
        


# policy(state,action)
policy_pi= {
    (0,0): 0, (0,1): 0, (0,2): 1,(0,3): 0,
    (1,0): 0, (1,1): 0, (1,2): 1,(1,3): 0,
    (2,0): 0, (2,1): 0, (2,2): 1,(2,3): 0,
    (4,0): 0, (4,1): 0, (4,2): 0,(4,3): 1,
    (6,0): 0, (6,1): 0, (6,2): 1,(6,3): 0,
    (8,0): 0, (8,1): 0, (8,2): 0,(8,3): 1,
    (9,0): 0, (9,1): 0, (9,2): 1,(9,3): 0,
    (10,0): 0,(10,1): 0,(10,2): 0,(10,3): 1,
    (11,0): 0,(11,1): 0,(11,2): 0,(11,3): 1,
}

v = {
    (0): 0,
    (1): 0,
    (2): 0,
    (3): 0,
    (4): 0,
    (5): 0,
    (6): 0,
    (7): 0,
    (8): 0,
    (9): 0,
    (10): 0,
    (11): 0, 
} 

########################## Parameters #####################
theta=0.0001
delta = float("inf")
# enviroment
EnviGridWorld3x4 = GridWorld3x4()

P=EnviGridWorld3x4.env.unwrapped.P    

##########################################################


###### Plot grild world 3*4 #####
fig = go.Figure()

fig.add_shape(type="rect",x0=0,y0=0,x1=1,y1=1,line=dict(color="black", width=3))
fig.add_shape(type="rect",x0=0,y0=1,x1=1,y1=2,line=dict(color="black", width=3))
fig.add_shape(type="rect",x0=0,y0=2,x1=1,y1=3,line=dict(color="black", width=3))
#
fig.add_shape(type="rect",x0=1,y0=0,x1=2,y1=1,line=dict(color="black", width=3))
fig.add_shape(type="rect",x0=1,y0=1,x1=2,y1=2,line=dict(color="black", width=3)
              ,fillcolor="black",)
fig.add_shape(type="rect",x0=1,y0=2,x1=2,y1=3,line=dict(color="black", width=3),)
#
fig.add_shape(type="rect",x0=2,y0=0,x1=3,y1=1,line=dict(color="black", width=3))
fig.add_shape(type="rect",x0=2,y0=1,x1=3,y1=2,line=dict(color="black", width=3))
fig.add_shape(type="rect",x0=2,y0=2,x1=3,y1=3,line=dict(color="black", width=3))
#
fig.add_shape(type="rect",x0=3,y0=0,x1=4,y1=1,line=dict(color="black", width=3))
fig.add_shape(type="rect",x0=3,y0=1,x1=4,y1=2,line=dict(color="black", width=3),
              fillcolor="red")
fig.add_shape(type="rect",x0=3,y0=2,x1=4,y1=3,line=dict(color="black", width=3),
              fillcolor="green")

# for text states
k=0
for i in [3,2,1]:
    for ii in [0,1,2,3]:
        fig.add_annotation(x=ii+ 0.15,y=i- 0.1,text=f"s{k}",showarrow=False,font=dict(size=14))
        k=k+1
    
##
fig.update_xaxes(range=[0, 4],showticklabels=False,showgrid=False,
                 zeroline=False,scaleanchor="y",scaleratio=1)
fig.update_yaxes(range=[0, 3],showticklabels=False,showgrid=False,
                 zeroline=False)
fig.update_layout(plot_bgcolor="white",paper_bgcolor="white")


# fig.add_annotation(x=0.4, y=0.5,text=str(v[(0)]),
#                    showarrow=False,font=dict(size=16),
#                        name="value_s8")
### for text value state
# row 0
fig.add_annotation(x=0.48, y=2.5,text=f"v(s0)={v[(0)]:.5f}",showarrow=False,font=dict(size=14),
                   name="value_s0")
fig.add_annotation(x=1.48, y=2.5,text=f"v(s1)={v[(1)]:.5f}",showarrow=False,font=dict(size=14),
                   name="value_s1")
fig.add_annotation(x=2.48, y=2.5,text=f"v(s2)={v[(2)]:.5f}",showarrow=False,font=dict(size=14),
                   name="value_s2")
fig.add_annotation(x=3.48, y=2.5,text=f"v(s3)={v[(3)]:.5f}",showarrow=False,font=dict(size=14),
                   name="value_s3")
# row 1
fig.add_annotation(x=0.48, y=1.5,text=f"v(s4)={v[(4)]:.5f}",showarrow=False,font=dict(size=14),
                   name="value_s4")
fig.add_annotation(x=1.48, y=1.5,text=f"v(s5)={v[(5)]:.5f}",showarrow=False,font=dict(size=14),
                   name="value_s5")
fig.add_annotation(x=2.48, y=1.5,text=f"v(s6)={v[(6)]:.5f}",showarrow=False,font=dict(size=14),
                   name="value_s6")
fig.add_annotation(x=3.48, y=1.5,text=f"v(s7)={v[(7)]:.5f}",showarrow=False,font=dict(size=14),
                   name="value_s7")
# row 2
fig.add_annotation(x=0.48, y=0.5,text=f"v(s8)={v[(8)]:.5f}",showarrow=False,font=dict(size=14),
                   name="value_s8")
fig.add_annotation(x=1.48, y=0.5,text=f"v(s9)={v[(9)]:.5f}",showarrow=False,font=dict(size=14),
                   name="value_s9")
fig.add_annotation(x=2.49, y=0.5,text=f"v(s10)={v[(10)]:.5f}",showarrow=False,font=dict(size=14),
                   name="value_s10")
fig.add_annotation(x=3.49, y=0.5,text=f"v(s11)={v[(11)]:.5f}",showarrow=False,font=dict(size=14),
                   name="value_s11")


#####################
app = Dash()

app.layout = html.Div([
    dcc.Graph(
        id='output1_graph',
        figure=fig,  
    style={
        'width': '99vw',
        'height': '94vh',
        'margin': 'auto'
    }
    ),
    
    html.Div(
        dcc.Input(
            id='input1_number',
            type='number',
            placeholder='Enter a number',
            value=0.01

        ),
        style={
            "display": "flex",
            "justifyContent": "center",
            "position": "fixed",
            "bottom": "55px",
            "width": "100%"
        }
    ),
    
    html.Div(
        html.Button("Enter", n_clicks=0, id="input2_button_enter", style={
                "fontSize": "20px",
                "padding": "10px 25px"}),
        style={
            "display": "flex",
            "justifyContent": "center",
            "position": "fixed",
            "bottom": "3px",
            "width": "100%"
        }
        ),
    
    
    # sleep
    dcc.Interval(
    id="input3_sleep",
    interval=1000,      # 1000 ms = 1 second
    n_intervals=0,
    disabled=True
                ),
    
    dcc.Store(
            id="delta_store",
            data=1e10
        )
            

])

@callback(
    
    Output('output1_graph', 'figure'),
    Output('input3_sleep', 'disabled'),
    Output("delta_store", "data"),

    Input('input2_button_enter', 'n_clicks'),
    State("input1_number", "value"),
    Input("input3_sleep", "n_intervals"), # input3
    State("delta_store", "data"), # input4


)


def function(input2_button_enter,input1_number,input3_sleep,delta_store):

    
    # when button click
    if ctx.triggered_id == "input2_button_enter":
        
        # If the input2_number box does not contain a value
        if input1_number==None:
            disabled=True
    
        else:
            disabled=False

        return (no_update,disabled,no_update)


    # when sleep
    if ctx.triggered_id == "input3_sleep":
        theta=input1_number
        delta = delta_store
        
        print(delta)
        
        if delta >  theta:  # k: Gauss-seidel method    
            delta=0
            for s in [0,1,2,4,6,8,9,10,11]:
                
                v_old= v[s]
    
                v_new=0
                for a in [0,1,2,3]:
                    
                    for prob, s_prime, reward, done in P[s][a]:
                        v_new= v_new+policy_pi[(s,a)]*prob*(reward+v[(s_prime)])
                        
                v[(s)]=v_new
                
                # Maximum change in V(s) across all states
                delta = max(delta, abs(v_old - v_new))
    
                # print( f"v[({s})]",v[(s)]) 
                
            # show value state
            fig.update_annotations(selector=dict(name="value_s0"),text=f"v(s0)={v[(0)]:.5f}")
            fig.update_annotations(selector=dict(name="value_s1"),text=f"v(s1)={v[(1)]:.5f}")
            fig.update_annotations(selector=dict(name="value_s2"),text=f"v(s2)={v[(2)]:.5f}")
            fig.update_annotations(selector=dict(name="value_s3"),text=f"v(s3)={v[(3)]:.5f}")

            fig.update_annotations(selector=dict(name="value_s4"),text=f"v(s4)={v[(4)]:.5f}")
            fig.update_annotations(selector=dict(name="value_s5"),text=f"v(s5)={v[(5)]:.5f}")
            fig.update_annotations(selector=dict(name="value_s6"),text=f"v(s6)={v[(6)]:.5f}")
            fig.update_annotations(selector=dict(name="value_s7"),text=f"v(s7)={v[(7)]:.5f}")

            fig.update_annotations(selector=dict(name="value_s8"),text=f"v(s8)={v[(8)]:.5f}")
            fig.update_annotations(selector=dict(name="value_s9"),text=f"v(s9)={v[(9)]:.5f}")
            fig.update_annotations(selector=dict(name="value_s10"),text=f"v(s10)={v[(10)]:.5f}")
            fig.update_annotations(selector=dict(name="value_s11"),text=f"v(s11)={v[(11)]:.5f}")


            disabled=False

        else:
            disabled=True
      
                
        
        
        print(input3_sleep)

        output1_graph=fig        

        return (output1_graph,disabled,delta)



    return no_update,no_update,no_update

if __name__ == '__main__':
    app.run(
        debug=False,
        port=8051
    )
