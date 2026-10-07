import plotly.graph_objects as go

fig = go.Figure()

fig.add_shape(type="rect",x0=0,y0=0,x1=1,y1=1,line=dict(color="black", width=3))
fig.add_shape(type="rect",x0=0,y0=1,x1=1,y1=2,line=dict(color="black", width=3))
fig.add_shape(type="rect",x0=0,y0=2,x1=1,y1=3,line=dict(color="black", width=3))


#
fig.add_shape(type="rect",x0=1,y0=0,x1=2,y1=1,line=dict(color="black", width=3))
fig.add_shape(type="rect",x0=1,y0=1,x1=2,y1=2,line=dict(color="black", width=3)
              ,fillcolor="black",)
fig.add_shape(type="rect",x0=1,y0=2,x1=2,y1=3,line=dict(color="black", width=3))

#
fig.add_shape(type="rect",x0=2,y0=0,x1=3,y1=1,line=dict(color="black", width=3))
fig.add_shape(type="rect",x0=2,y0=1,x1=3,y1=2,line=dict(color="black", width=3))
fig.add_shape(type="rect",x0=2,y0=2,x1=3,y1=3,line=dict(color="black", width=3))

#
fig.add_shape(type="rect",x0=3,y0=0,x1=4,y1=1,line=dict(color="black", width=3))
fig.add_shape(type="rect",x0=3,y0=1,x1=4,y1=2,line=dict(color="black", width=3))
fig.add_shape(type="rect",x0=3,y0=2,x1=4,y1=3,line=dict(color="black", width=3))

k=0
for i in [3,2,1]:
    for ii in [0,1,2,3]:
        fig.add_annotation(x=ii+ 0.15,y=i- 0.1,text=f"s{k}",showarrow=False,font=dict(size=14))
        k=k+1
    
# for s in range(12):
#     row = s // 4
#     col = s % 4

#     fig.add_annotation(
#         x=col + 0.1,
#         y=2.9 - row,
#         text=str(s),
#         showarrow=False,
#         font=dict(size=14)
#     )
    



fig.update_xaxes(range=[0, 4],showticklabels=False,showgrid=False,
                 zeroline=False,scaleanchor="y",scaleratio=1)

fig.update_yaxes(range=[0, 3],showticklabels=False,showgrid=False,
                 zeroline=False)

fig.update_layout(plot_bgcolor="white",paper_bgcolor="white")


fig.write_html('first_figure1.html', auto_open=True)