# save this as app.py
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html

# Data
df = px.data.gapminder().query("year==2007")

df = df.rename(columns=dict(pop="Population",
                            gdpPercap="GDP per Capita",
                            lifeExp="Life Expectancy"))

cols_dd = ["Population", "GDP per Capita", "Life Expectancy"]

app = dash.Dash()
app.layout = html.Div([
    dcc.Dropdown(
        id='demo-dropdown',
        options=[{'label': k, 'value': k} for k in cols_dd],
        value=cols_dd[0]
    ),

    html.Hr(),
    dcc.Graph(id='display-selected-values'),

])

@app.callback(
    dash.dependencies.Output('display-selected-values', 'figure'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
    
def update_output(value):
    fig = go.Figure()
    fig.add_trace(go.Choropleth(
       locations=df['iso_alpha'], # Spatial coordinates
        z=df[value].astype(float), # Data to be color-coded
        colorbar_title=value))
    fig.update_layout(title=f"<b>{value}</b>", title_x=0.5)
    return fig

if __name__ == '__main__':
    app.run_server()