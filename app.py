# Import numpy and pandas and read dataset
import numpy as np 
import pandas as pd 
# Import preprocessing tools from sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
# Dash related components
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
# Plotly components
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
# Load utilities
from utilities import *


colorpalette=['#f6b48e','#e13242','#35183d','#6f1f57','#ac1759'
              ,'#f37651','red']

dataset = Dataset()
corr_matrix = dataset.compute_correlation()

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.FLATLY])
server = app.server

GITHUB_LOGO = "https://raw.githubusercontent.com/fmani/dash_board-stroke-prediction/main/Logos/GitHub_Logo_White.png"
navbar = dbc.Navbar(
    [
       html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=GITHUB_LOGO, height="50px")),
                    dbc.Col(dbc.NavbarBrand("Link to GitHub repo", className="ml-2")),
                    
                ],
                align="center",
                no_gutters=True,
            ),
           href="https://github.com/fmani/dash_board-stroke-prediction",
        ),
        
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=GITHUB_LOGO, height="50px")),
                    dbc.Col(dbc.NavbarBrand("Link to Kaggle dataset", className="ml-2")),
                    
                ],
                align="center",
                no_gutters=True,
            ),
           href="https://www.kaggle.com/fedesoriano/stroke-prediction-dataset",
        ),


        html.H1(
            children='Stroke Risk Factors',
            style={
                'textAlign': 'right',
                'color': 'white',
                'margin-left': '300px'
            }),


    ],
    color='#325899',
    dark=True,
)

description_row = html.Div(children=[
    dcc.Markdown('''
    ### Analysis of the stroke risk dataset
    Interactive dashboard allowing to study different aspects which may influence the stroke risk factor.  Instructions:
    * Select the couple of features by selecting the correlation matrix element in the matrix on the right
    * The two plots on the left show the distributions of the selected features, grouped by stroke incidence.
    
    ''')],
    style={
        'padding': '15px 30px 27px',
        'margin-left': '20px',
        'margin-up': '20px',
        'width': '80%',
        'max-width': '1024px',
        'font-family': 'sans-serif'
    },
)

dropdown_menu = html.Div(children=[html.P("Select Admitted Features"),
                                   dcc.Dropdown(
                                       id="admit-select",
                                       options=[{"label": dataset.nice_labels[i], "value": i} for i in dataset.columns],
                                       value=dataset.columns[:],
                                       multi=True,
                                   )])


app.layout = html.Div(
    [
        dbc.Row(dbc.Col(navbar,)),
        dbc.Row([dbc.Col(description_row,width=5),
                 dbc.Col(dropdown_menu,width=5)]),
        dbc.Row([

            dbc.Col(
                html.Div(children = [
                    dcc.Graph(id='distribution_var_1')]),
                width=4),
            dbc.Col(html.Div(
                children = [dcc.Graph(id="correlation_hm")]
            ),width=8), 
            ]),
        dbc.Row(
            [
                dbc.Col(
                html.Div(children = [
                    dcc.Graph(id='distribution_var_2')]),
                    width=4),
                dbc.Col(
                html.Div(children = [
                    dcc.Graph(id='scatter_plot')]
                ), width=7),
            ],
            
        ),
    ]
)


def generate_density(hm_click,which,admit_feats):

    x_axis = list(dataset.columns)
    y_axis = list(dataset.columns)
    
    feat = admit_feats[which]

    if hm_click is not None:
        ind = "x" if which==0 else "y"
        feat = hm_click["points"][0][ind] 
        

    data = []
    color_counter = 0
    for a,b in dataset.df.groupby('stroke'):
        data.append(dict( x = b[feat], type='histogram',
                          histnorm='probability density',
                          opacity=0.75,
                          name={1:'Stroke: Yes',0:'Stroke: No'}[a],
                          labels=b[feat].unique(),
                          marker=dict(color=colorpalette[color_counter])))
        color_counter +=1

    layout =  {
        'xaxis': {'side':'bottom','autorange':True,
                  'automargin': True, 'animate': True,},
        'yaxis': {'autorange':True,'automargin': True, 'animate': True,
                  'title':'Density'},
        'barmode':'overlay',
        'font':dict( family="sans-serif",
                     size=18,
                     color="Black"),
        'title':dataset.nice_labels[feat]+' distribution',
    }
    
    if feat in dataset.categ_feats:
        feat_ind = dataset.categ_feats.index(feat)
        layout['xaxis']['tickmode']='array'
        layout['xaxis']['ticktext']=dataset.categories_decode[feat_ind]

    if feat in ['hypertension','heart_disease','stroke']:
        layout['xaxis']['tickmode']='array'
        layout['xaxis']['ticktext']=['No','Yes']
        

    return {"data": data, "layout": layout}



def generate_scatter_plot(hm_click,admit_feats):

    x_axis = list(dataset.columns)
    y_axis = list(dataset.columns)
    
    feat_x = admit_feats[0]
    feat_y = admit_feats[1]

    if hm_click is not None:
        feat_x = hm_click["points"][0]['x'] 
        feat_y = hm_click["points"][0]['y']

    if(feat_x not in admit_feats):
        feat_x = admit_feats[0]
    if(feat_y not in admit_feats):
        feat_y = admit_feats[1]
  
        
    data = []

    is_categ = (feat_x in dataset.categ_feats) or (feat_y in dataset.categ_feats)

    
    if(is_categ):
        color_counter=0
        slow_feat = feat_x if (feat_x in dataset.categ_feats) else feat_y
        fast_feat = feat_x if (slow_feat==feat_y) else feat_y
        for a,b in dataset.df.groupby(slow_feat):
            data.append(dict(x = b[fast_feat], type='histogram',
                             name=a,
                             opacity=0.75,histnorm='probability density',
                             marker=dict(color=colorpalette[color_counter])))
            color_counter += 1
    else:
        color_counter=0
        for a,b in dataset.df.groupby('stroke'):
            data.append(dict(x = b[feat_x], y=b[feat_y], type='scatter',
                             opacity=0.75,
                             name={1:'Stroke: Yes',0:'Stroke: No'}[a],
                             mode='markers',
                             marker=dict(color=colorpalette[color_counter])))
            color_counter += 1

    layout =  {
        'xaxis': {'side':'bottom','autorange':True,
                  'automargin': True, 'animate': True,
                  'title':dataset.nice_labels[feat_x]},
        'yaxis': {'autorange':True,'automargin': True,
                  'animate': True,'title':'Density' if is_categ
                  else dataset.nice_labels[feat_y]},
        'barmode':'overlay',
        'font':dict( family="sans-serif",
                     size=18,
                     color="Black"),
        'title':dataset.nice_labels[feat_x]+' vs.' +dataset.nice_labels[feat_y],
        'annotations' : [dict(yref='paper',xref="paper",
                              y=1.1,x=1.05,
                              text=dataset.nice_labels[slow_feat]+' :' if is_categ else ''
                              ,showarrow=False)]
    }

       
    for i,feat in enumerate([feat_x,feat_y]):
            if feat in dataset.categ_feats: 
                feat_ind = dataset.categ_feats.index(feat)
                layout[['xaxis','yaxis'][i]]['tickmode']='array'
                layout[['xaxis','yaxis'][i]]['ticktext']=dataset.categories_decode[feat_ind]
                
        
    return {"data": data, "layout": layout}


        

def generate_correlation_heatmap(hm_click,admit_feats):

    x_axis = list(dataset.columns)
    y_axis = list(dataset.columns)
    
    feat_x = admit_feats[0]
    feat_y = admit_feats[1]

    
    z = corr_matrix.copy()
    data = [
        dict(
            x=x_axis,
            y=y_axis,
            z=z,
            type="heatmap",
            name="",
            showscale=True,
            colorscale='Portland',
            colorbar=dict(title='Correlation')
        )
    ]

    annotation = dataset.get_annotations()
    if hm_click is not None:
        feat_x = hm_click["points"][0]["x"]
        feat_y = hm_click["points"][0]["y"]

    if(feat_x not in admit_feats):
        feat_x = admit_feats[0]
    if(feat_y not in admit_feats):
        feat_y = admit_feats[1]
    
    idx_x = dataset.columns.index(feat_x)
    idx_y = dataset.columns.index(feat_y)
        
    annotation[dataset.n_cols*idx_x+idx_y].update(size=15,
                                                  font=dict(color='red'))
        
    layout =  {
        'xaxis': {'side':'top','autorange':True,'automargin': True, 'animate': True},
        'yaxis': {'autorange':True,'automargin': True, 'animate': True},
        'font':dict( family="sans-serif",
                     size=18,
                     color="Black"),
        'annotations':annotation,
    }
    
    
        
    
    return {"data": data, "layout": layout}
        
        
@app.callback(
    Output("correlation_hm", "figure"),
    [
        Input("correlation_hm", "clickData"),
        Input("admit-select", "value"),
    ],
)
def update_heatmap(hm_click,admit_feats):
    dataset.update_features(admit_feats)
    corr_matrix = dataset.compute_correlation()
    
    return generate_correlation_heatmap(hm_click,admit_feats)    


@app.callback(
    Output("scatter_plot", "figure"),
    [
        Input("correlation_hm", "clickData"),
        Input("admit-select", "value"),
    ],
)
def update_heatmap(hm_click,admit_feats):
    return generate_scatter_plot(hm_click,admit_feats)    




@app.callback(
    [
        Output("distribution_var_1", "figure"),
        Output("distribution_var_2", "figure")
    ],
    [
        Input("correlation_hm", "clickData"),
        Input("admit-select", "value")
    ],
)
def update_distr_1(hm_click,admit_feats):
    return generate_density(hm_click,0,admit_feats),generate_density(hm_click,1,admit_feats)    


if __name__ == '__main__':
    app.run_server()







