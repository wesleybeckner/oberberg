# -*- coding: utf-8 -*-
import dash
import json
import dash_core_components as dcc
import dash_daq as daq
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np
import datetime

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

opp = pd.read_csv('data/opportunity.csv', index_col=[0,1,2,3])
opportunity = pd.read_csv('data/days.csv', index_col=[0,1,2,3])
annual_operating = pd.read_csv('data/annual.csv', index_col=[0,1])
stats = pd.read_csv('data/scores.csv')
quantiles = np.arange(50,101,1)
quantiles = quantiles*.01
quantiles = np.round(quantiles, decimals=2)
dataset = opp.sort_index()
lines = opp.index.get_level_values(1).unique()
asset_metrics = ['Yield', 'Rate', 'Uptime']
groupby = ['Line', 'Product group']
oee = pd.read_csv('data/oee.csv', index_col=0)
oee['From Date/Time'] = pd.to_datetime(oee["From Date/Time"])
oee['To Date/Time'] = pd.to_datetime(oee["To Date/Time"])
oee["Run Time"] = pd.to_timedelta(oee["Run Time"])
res = oee.groupby(groupby)[asset_metrics].quantile(quantiles)

def find_quantile(to_remove_line='E26', to_add_line='E27',
                  metrics=['Rate', 'Yield', 'Uptime'],
                  uptime=None):
    to_remove_kg = annual_operating.loc[to_remove_line]['Quantity Good']
    to_remove_rates = res.loc[to_remove_line].unstack()['Rate']
    to_remove_yields = res.loc[to_remove_line].unstack()['Yield']
    target_days_needed = pd.DataFrame(to_remove_kg).values / to_remove_yields / to_remove_rates / 24
    target_days_needed = target_days_needed.T
    target_days_needed['Total'] = target_days_needed.sum(axis=1)

    target_data = opportunity.loc['Additional Days', to_add_line].unstack()[metrics].sum(axis=1)
    target = pd.DataFrame(target_data).T
    target.index=['Days']
    target = target.T
    if uptime != None:
        target['Days'] = target['Days'] + uptime

    final = pd.merge(target_days_needed, target, left_index=True, right_index=True)
    quantile = (abs(final['Days'] - final['Total'])).idxmin()
    return quantile, final.iloc[:-1]

def make_consolidate_plot(remove='E26', add='E27',
                          metrics=['Rate', 'Yield', 'Uptime'],
                          uptime=None):
    quantile, final = find_quantile(remove, add, metrics, uptime)
    fig = go.Figure(data=[
    go.Bar(name='Days Available', x=final.index, y=final['Days']),
    go.Bar(name='Days Needed', x=final.index, y=final['Total'])
    ])
    # Change the bar mode
    fig.update_layout(barmode='group',
                  yaxis=dict(title="Days"),
                   xaxis=dict(title="Quantile"))
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
    })
    return fig

def pareto_product_family(quantile=0.9, clickData=None):
    if clickData != None:
        line = clickData["points"][0]["y"]
    else:
        line = 'K40'
    data = opportunity.reorder_levels([0,2,1,3]).loc['Additional Days', quantile, line]
    total = data.sum().sum()
    cols = data.columns
    bar_fig = []
    for col in cols:
        bar_fig.append(
        go.Bar(
        name=col,
        orientation="h",
        y=[str(i) for i in data.index],
        x=data[col],
        customdata=[col],
        )
        )

    figure = go.Figure(
        data=bar_fig,
        layout=dict(
            barmode="group",
            yaxis_type="category",
            yaxis=dict(title="Product Group"),
            xaxis=dict(title="Days"),
            title="{}: {:.1f} days of opportunity".format(line,total),
            plot_bgcolor="#F9F9F9",
            paper_bgcolor="#F9F9F9"
        )
    )
    figure.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
    })
    return figure

def make_days_plot(quantile=0.9):
    data = opportunity.reorder_levels([0,2,1,3]).sort_index().loc['Additional Days', quantile].groupby('Line').sum()
    cols = ['Rate', 'Yield', 'Uptime']
    data['Total'] = data.sum(axis=1)
    data = data.sort_values(by='Total')
    bar_fig = []
    for col in cols:
        bar_fig.append(
        go.Bar(
        name=col,
        orientation="h",
        y=[str(i) for i in data.index],
        x=data[col],
        customdata=[col]
        )
    )

    figure = go.Figure(
        data=bar_fig,
        layout=dict(
            barmode="stack",
            yaxis_type="category",
            yaxis=dict(title="Line"),
            xaxis=dict(title="Days"),
            title="Annualized Opportunity",
            plot_bgcolor="#F9F9F9",
            paper_bgcolor="#F9F9F9"
        )
    )
    figure.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
    })
    return figure

def make_culprits():
    fig = px.bar(stats, x='group', y='score', color='metric',
        barmode='group')
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
    })
    return fig

def pie_line(clickData=None):
    if clickData != None:
        line = clickData["points"][0]["y"]
    else:
        line = 'K40'
    data = annual_operating.loc[line]
    total = data['Net Quantity Produced'].sum()/1e6
    fig = px.pie(data, values='Net Quantity Produced', names=data.index, title='Production distribution 2019 ({:.1f}M kg)'.format(total))
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
    })
    return fig

def calculate_opportunity(quantile=0.9):
    data = opportunity.reorder_levels([0,2,1,3]).sort_index().loc['Additional Days', quantile].groupby('Line').sum()
    data['Total'] = data.sum(axis=1)
    return "{:.1f}".format(data.sum()[3]), \
            "{:.1f}".format(data.sum()[0]), \
            "{:.1f}".format(data.sum()[1]), \
            "{:.1f}".format(data.sum()[2])
# Describe the layout/ UI of the app
app.layout = html.Div([
    html.H4(["Untapped Potential"]),
    html.P("Opportunity (days of additional production) is computed from distributions around uptime, yield, and rate with respect to each of the lines and their product families. Some lines perform very well (E27 and K06) and already perform near their upper quantile ranges. Other lines (K10, E28) have a lot of hidden capacity due to wide variability in their operation. The additional days of production should be interpreted as untapped potential. For instance, If all lines were to perform in their 0.82 quantile bracket, the plant would gain the equivalent of running an additional line for an entire calendar year.  "),
    html.Div([
        html.Div([
            html.H6(id='new-rev'), html.P('Total Days of Production Saved')
        ], className='mini_container',
           id='rev',
        ),
        html.Div([
            html.H6(id='new-rev-percent'), html.P('Rate (days)')
        ], className='mini_container',
           id='rev-percent',
        ),
        html.Div([
            html.H6(id='new-products'), html.P('Yield (days)')
        ], className='mini_container',
           id='products',
        ),
        html.Div([
            html.H6(id='new-products-percent'), html.P('Uptime (days)')
        ], className='mini_container',
           id='products-percent',
        ),
    ], className='row container-display'
    ),
    html.Div([
        html.H6(id='slider-selection'),
        dcc.Slider(id='quantile_slider',
                    min=0.51,
                    max=0.99,
                    step=0.01,
                    value=.82,
                    included=False,
                    className="dcc_control"),
        dcc.Graph(
                    id='bar_plot',
                    figure=make_days_plot()),
            ], className='mini_container',
            ),
    html.Div([
        html.Div([
            dcc.Graph(
                        id='pareto_plot',
                        figure=pareto_product_family())
                ], className='mini_container',
                   id='pareto',
                ),
        html.Div([
            dcc.Graph(
                        id='pie_plot',
                        figure=pie_line())
                ], className='mini_container',
                   id='pie',
                ),
            ], className='row container-display',
            ),
    html.H4("Line Consolidation"),
    html.P("With the given line performances there is an opportunity for "\
            "consolidation. 'Days Needed' are computed from rate, yield and "\
            "the total production for 'Line to Remove' in 2019. "\
            "'Days Available' is computed from rate, yield, and uptime "\
            "improvements in 'Line to Overload'. A manual overide is "\
            "available to remove uptime consideration. In this case, uptime "\
            "can be manually inputed, with a maximum value based on the "\
            "downtime days for that line in 2019."),
    html.Div([
        html.Div([
            html.Div([
                html.P("Line to Remove"),
                dcc.Dropdown(id='line-in-selection',
                            options=[{'label': i, 'value': i} for i in \
                                     lines],
                            value='E26',),
                    ], className='mini_container',
                       id='line-in',
                    ),
            html.Div([
                html.P("Line to Overload"),
                dcc.Dropdown(id='line-out-selection',
                            options=[{'label': i, 'value': i} for i in \
                                     lines],
                            value='E27',),
                    ], className='mini_container',
                       id='line-out',
                    ),
            html.Div([
                html.P('Uptime manual overide'),
                daq.BooleanSwitch(
                  id='daq-switch',
                  on=False,
                  style={"margin-bottom": "10px"}),
                dcc.Slider(id='uptime-slider',
                            min=0,
                            max=10,
                            step=1,
                            value=9,
                            included=True,
                            className="dcc_control"),
                    ], className='mini_container',
                        id='switch',
                    ),
                ], className='row container-display',
                ),
            html.H6(id='quantile-target'),
        dcc.Graph(
                    id='consolidate_plot',
                    figure=make_consolidate_plot()),
            ], className='mini_container',
            ),
    html.H4("The Usual Suspects"),
    html.P("Scores reflect whether a group (line or product family) is "\
           "improving or degrading the indicated metric (uptime, rate, yield). "\
           "While groups were determined to be statistically impactful "\
           "(null hypothesis < 0.01) it does not guarantee decoupling. For "\
           "instance, PSL has a very negative impact on rate and yield. "\
           "However, the only line that runs PSL is E28, which is rated similarly."),
    html.Div([
        dcc.Graph(
                    id='scores_plot',
                    figure=make_culprits()),
        html.Pre(id='slider-data'),
        html.Pre(id='click-data'),
            ], className='mini_container',
            ),
    ], className='pretty container'
    )

app.config.suppress_callback_exceptions = False

@app.callback(
    [Output('new-rev', 'children'),
     Output('new-rev-percent', 'children'),
     Output('new-products', 'children'),
     Output('new-products-percent', 'children')],
    [Input('quantile_slider', 'value')]
)
def display_opportunity(quantile):
    return calculate_opportunity(quantile)

@app.callback(
    Output('uptime-slider', 'disabled'),
    [Input('daq-switch', 'on')])
def display_click_data(on):
    return on == False

@app.callback(
    Output('uptime-slider', 'max'),
    [Input('line-out-selection', 'value')])
def display_click_data(line):
    days = np.round(oee.loc[oee['Line'] == line]['Uptime'].sum()/24)
    return days

@app.callback(
    Output('bar_plot', 'figure'),
    [Input('quantile_slider', 'value')])
def display_click_data(quantile):
    return make_days_plot(quantile)

@app.callback(
    Output('consolidate_plot', 'figure'),
    [Input('line-in-selection', 'value'),
     Input('line-out-selection', 'value'),
     Input('daq-switch', 'on'),
     Input('uptime-slider', 'value')]
     )
def display_click_data(inline, outline, switch, uptime):
    if switch == True:
        return make_consolidate_plot(inline, outline, ['Rate', 'Yield'], uptime)
    else:
        return make_consolidate_plot(inline, outline)

@app.callback(
    Output('quantile-target', 'children'),
    [Input('line-in-selection', 'value'),
     Input('line-out-selection', 'value'),
     Input('daq-switch', 'on'),
     Input('uptime-slider', 'value')]
     )
def display_click_data(inline, outline, switch, uptime):
    if switch == True:
        quantile, final = find_quantile(inline, outline, ['Rate', 'Yield'],
                    uptime)
        return "Quantile-Performance Target: {} + {} Uptime Days"\
            .format(quantile, uptime)
    else:
        quantile, final = find_quantile(inline, outline)
        return "Quantile-Performance Target: {}".format(quantile)

@app.callback(
    Output('pareto_plot', 'figure'),
    [Input('quantile_slider', 'value'),
     Input('bar_plot', 'clickData')])
def display_click_data(quantile, clickData):
    return pareto_product_family(quantile, clickData)

@app.callback(
    Output('pie_plot', 'figure'),
    [Input('bar_plot', 'clickData')])
def display_click_data(clickData):
    return pie_line(clickData)

@app.callback(
    Output('slider-selection', 'children'),
    [Input('quantile_slider', 'value')])
def display_click_data(quantile):
    return "Quantile: {}".format(quantile)

if __name__ == "__main__":
    app.run_server(debug=True)
