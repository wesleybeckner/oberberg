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
from itertools import cycle

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
oee = pd.read_csv('data/oee.csv')
oee['From Date/Time'] = pd.to_datetime(oee["From Date/Time"])
oee['To Date/Time'] = pd.to_datetime(oee["To Date/Time"])
oee["Run Time"] = pd.to_timedelta(oee["Run Time"])
oee = oee.loc[oee['Rate'] < 2500]
res = oee.groupby(groupby)[asset_metrics].quantile(quantiles)

df = pd.read_csv('data/products.csv')
descriptors = df.columns[:8]
production_df = df
production_df['product'] = production_df[descriptors[2:]].agg('-'.join, axis=1)
production_df = production_df.sort_values(['Product Family', 'EBIT'], ascending=False)

stat_df = pd.read_csv('data/category_stats.csv')
old_products = df[descriptors].sum(axis=1).unique().shape[0]

def calculate_margin_opportunity(sort='Worst', select=[0,10], descriptors=None):
    if sort == 'Best':
        local_df = stat_df.sort_values('score', ascending=False)
        local_df = local_df.reset_index(drop=True)
    else:
        local_df = stat_df
    if descriptors != None:
        local_df = local_df.loc[local_df['descriptor'].isin(descriptors)]
    if sort == 'Best':

        new_df = pd.DataFrame()
        for index in range(select[0],select[1]):
            x = df.loc[(df[local_df.iloc[index]['descriptor']] == \
                local_df.iloc[index]['group'])]
            new_df = pd.concat([new_df, x])
    else:

        new_df = df
        for index in range(select[0],select[1]):
            new_df = new_df.loc[~(new_df[local_df.iloc[index]['descriptor']] ==\
                    local_df.iloc[index]['group'])]

    new_EBIT = 1 / (new_df['Sales Quantity in KG'].sum() /
        df['Sales Quantity in KG'].sum()) * new_df['EBIT'].sum()

    EBIT_percent = (new_df['EBIT'].sum()) / df['EBIT'].sum() * 100
    new_products = new_df[descriptors].sum(axis=1).unique().shape[0]
    product_percent_reduction = (new_products) / \
        old_products * 100
    new_kg = new_df['Sales Quantity in KG'].sum()
    old_kg = df['Sales Quantity in KG'].sum()
    kg_percent = new_kg / old_kg * 100

    return "${:.1f} M of ${:.1f} M ({:.1f}%)".format(new_df['EBIT'].sum()/1e6, df['EBIT'].sum()/1e6, EBIT_percent), \
            "{} of {} Products ({:.1f}%)".format(new_products,old_products,product_percent_reduction),\
            "{:.1f} M of {:.1f} M kg ({:.1f}%)".format(new_kg/1e6, old_kg/1e6, kg_percent)

def make_violin_plot(sort='Worst', select=[0,10], descriptors=None):

    if sort == 'Best':
        local_df = stat_df.sort_values('score', ascending=False)
        local_df = local_df.reset_index(drop=True)
    else:
        local_df = stat_df
    if descriptors != None:
        local_df = local_df.loc[local_df['descriptor'].isin(descriptors)]
    fig = go.Figure()
    for index in range(select[0],select[1]):
        x = df.loc[(df[local_df.iloc[index]['descriptor']] == \
            local_df.iloc[index]['group'])]['EBIT']
        y = local_df.iloc[index]['descriptor'] + ': ' + df.loc[(df[local_df.iloc\
            [index]['descriptor']] == local_df.iloc[index]['group'])]\
            [local_df.iloc[index]['descriptor']]
        name = 'EBIT: {:.0f}, {}'.format(x.median(),
            local_df.iloc[index]['group'])
        fig.add_trace(go.Violin(x=y,
                                y=x,
                                name=name,
                                box_visible=True,
                                meanline_visible=True))
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
                "title": 'EBIT by Product Descriptor',
                })

    return fig

def make_sunburst_plot(clickData=None, toAdd=None, col=None, val=None):
    if clickData != None:
        col = clickData["points"][0]['x'].split(": ")[0]
        val = clickData["points"][0]['x'].split(": ")[1]
    elif col == None:
        col = 'Thickness Material A'
        val = '47'

    desc = list(descriptors[:-2])
    if col in desc:
        desc.remove(col)
    if toAdd != None:
        for item in toAdd:
            desc.append(item)
    test = production_df.loc[production_df[col] == val]
    fig = px.sunburst(test, path=desc[:], color='EBIT', title='{}: {}'.format(
        col, val),
        color_continuous_scale='RdBu')
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
                "title": 'EBIT, {}: {}'.format(col,val),
                })
    return fig

def make_ebit_plot(production_df, select=None, sort='Worst', descriptors=None):
    families = production_df['Product Family'].unique()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    colors_cycle = cycle(colors)
    grey = ['#7f7f7f']
    color_dic = {'{}'.format(i): '{}'.format(j) for i, j  in zip(families, colors)}
    grey_dic =  {'{}'.format(i): '{}'.format('#7f7f7f') for i in families}
    fig = go.Figure()


    if select == None:
        for data in px.scatter(
                production_df,
                x='product',
                y='EBIT',
                color='Product Family',
                color_discrete_map=color_dic,
                opacity=1).data:
            fig.add_trace(
                data
            )

    elif select != None:
        color_dic = {'{}'.format(i): '{}'.format(j) for i, j  in zip(select, colors)}
        for data in px.scatter(
                production_df,
                x='product',
                y='EBIT',
                color='Product Family',

                color_discrete_map=color_dic,
                opacity=0.09).data:
            fig.add_trace(
                data
            )

        if sort == 'Best':
            local_df = stat_df.sort_values('score', ascending=False)
        elif sort == 'Worst':
            local_df = stat_df


        new_df = pd.DataFrame()
        if descriptors != None:
            local_df = local_df.loc[local_df['descriptor'].isin(descriptors)]
        for index in select:
            x = production_df.loc[(production_df[local_df.iloc[index]['descriptor']] == \
                local_df.iloc[index]['group'])]
            x['color'] = next(colors_cycle) # for line shapes
            new_df = pd.concat([new_df, x])
            new_df = new_df.reset_index(drop=True)
        for data in px.scatter(
                new_df,
                x='product',
                y='EBIT',
                color='Product Family',

                color_discrete_map=color_dic,
                opacity=1).data:
            fig.add_trace(
                data
            )
        shapes=[]

        for index, i in enumerate(new_df['product']):
            shapes.append({'type': 'line',
                           'xref': 'x',
                           'yref': 'y',
                           'x0': i,
                           'y0': -4e5,
                           'x1': i,
                           'y1': 4e5,
                           'line':dict(
                               dash="dot",
                               color=new_df['color'][index],)})
                               # color=color_dic[new_df['Product Family'][index]])})
        fig.update_layout(shapes=shapes)
    fig.update_layout({
            "plot_bgcolor": "#F9F9F9",
            "paper_bgcolor": "#F9F9F9",
            "title": 'EBIT by Product Family',
            "height": 750,
            })
    return fig

def calculate_overlap(lines=['E27', 'E26']):
    path=['Product group', 'Polymer', 'Base Type', 'Additional Treatment']

    line1 = oee.loc[oee['Line'].isin([lines[0]])].groupby(path)['Quantity Good'].sum()
    line2 = oee.loc[oee['Line'].isin([lines[1]])].groupby(path)['Quantity Good'].sum()

    set1 = set(line1.index)
    set2 = set(line2.index)

    both = set1.intersection(set2)
    unique = set1.union(set2) - both

    kg_overlap = (line1.loc[list(both)].sum() + line2.loc[list(both)].sum()) /\
    (line1.sum() + line2.sum())
    return kg_overlap*100

def make_product_sunburst(lines=['E27', 'E26']):
    fig = px.sunburst(oee.loc[oee['Line'].isin(lines)],
        path=['Product group', 'Polymer', 'Base Type', 'Additional Treatment', 'Line'],
        color='Line')
    overlap = calculate_overlap(lines)
    fig.update_layout({
                 "plot_bgcolor": "#F9F9F9",
                 "paper_bgcolor": "#F9F9F9",
                 "height": 500,
                 "margin": dict(
                        l=0,
                        r=0,
                        b=0,
                        t=30,
                        pad=4
    ),
                 "title": "Product Overlap {:.1f}%: {}, {}".format(overlap, lines[0], lines[1]),
     })
    return fig

def make_metric_plot(line='K40', pareto='Product', marginal='rug'):
    plot = oee.loc[oee['Line'] == line]
    plot = plot.sort_values('Thickness Material A')
    plot['Thickness Material A'] = pd.to_numeric(plot['Thickness Material A'])
    fig = px.density_contour(plot, x='Rate', y='Yield',
                 color=pareto, marginal_x=marginal, marginal_y=marginal)
    fig.update_layout({
                 "plot_bgcolor": "#F9F9F9",
                 "paper_bgcolor": "#F9F9F9",
                 "height": 750,
                 "title": "{}, Pareto by {}".format(line, pareto),
     })
    return fig

def make_utilization_plot():
    downdays = pd.DataFrame(oee.groupby('Line')['Uptime'].sum().sort_values()/24)
    downdays.columns = ['Unutilized Days, 2019']
    fig = px.bar(downdays, y=downdays.index, x='Unutilized Days, 2019',
           orientation='h', color=downdays.index)
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
                "title": "Utilization, All Lines",
                "height": 300,
    })
    return fig

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
    if uptime != None:
        title = "Quantile-Performance Target: {} + {} Uptime Days"\
            .format(quantile, uptime)
    else:
        title = "Quantile-Performance Target: {}".format(quantile)
    # Change the bar mode
    fig.update_layout(barmode='group',
                  yaxis=dict(title="Days"),
                   xaxis=dict(title="Quantile"))
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
                "title": title,
                "margin": dict(
                       l=0,
                       r=0,
                       b=0,
                       t=30,
                       pad=4
   ),
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
    html.H4(["Product Characterization"]),
    html.P("Product descriptors are sorted by best or worst EBIT medians. Selecting these descriptors automatically computes annualized EBIT"),
    html.P("Use case I: If Gendorf were to ONLY produce products described by the top 10 descriptors they would increase their EBIT by 430% and reduce their product portfolio by 93%. Conversely, in eliminating the worst 10 descriptors EBIT would increase by 99% and the product portfolio would be reduced by 23%."),
    html.P("Use Case II: I see that Base Type 459/01 is a 'Good' product descriptor. However I'm curious if it describes a single product family. I select this descriptor in the violin plot to update the sunburst plot, I see that this describes Cooling Tower, Other Technical, and Construction products. The sunburst plot also tells me that the majority of this EBIT gain is focused in the Coolin Tower products. If I use the range slider in the violin plot to only include this descriptor in the annualized EBIT calculation I see that to construct a product portfolio with only these products I would increase EBIT by 70%"),
    html.Div([
        html.Div([
            html.H6(id='margin-new-rev'), html.P('EBIT')
        ], className='mini_container',
           id='margin-rev',
        ),
        html.Div([
            html.H6(id='margin-new-rev-percent'), html.P('Unique Products')
        ], className='mini_container',
           id='margin-rev-percent',
        ),
        html.Div([
            html.H6(id='margin-new-products'), html.P('Volume')
        ], className='mini_container',
           id='margin-products',
        ),
    ], className='row container-display'
    ),
    html.Div([
        # html.P('Sort product descriptors by selecting (best) products for '\
        #     'portfolio or eliminating (worst) products from portfolio'),
        html.Div([
            html.P('Descriptors'),
            dcc.Dropdown(id='descriptor_dropdown',
                         options=[{'label': 'Thickness', 'value': 'Thickness Material A'},
                                 {'label': 'Width', 'value': 'Width Material Attri'},
                                 {'label': 'Base Type', 'value': 'Base Type'},
                                 {'label': 'Additional Treatment', 'value': 'Additional Treatment'},
                                 {'label': 'Color', 'value': 'Color Group'},
                                 {'label': 'Product Group', 'value': 'Product Group'},
                                 {'label': 'Base Polymer', 'value': 'Base Polymer'},
                                 {'label': 'Product Family', 'value': 'Product Family'}],
                         value=['Thickness Material A',
                                'Width Material Attri', 'Base Type',
                                'Additional Treatment', 'Color Group',
                                'Product Group',
                                'Base Polymer', 'Product Family'],
                         multi=True,
                         className="dcc_control"),
            html.P('Number of Descriptors:', id='descriptor-number'),
            dcc.RangeSlider(
                        id='select',
                        min=0,
                        max=stat_df.shape[0],
                        step=1,
                        value=[0,10],
            ),
            html.P('Sort by:'),
            dcc.RadioItems(
                        id='sort',
                        options=[{'label': i, 'value': i} for i in \
                                ['Best', 'Worst']],
                        value='Best',
                        labelStyle={'display': 'inline-block'},
                        style={"margin-bottom": "10px"},),
                ], className='mini_container',
                    id='descriptorBlock',
                ),
        ], className='row container-display',
        ),
    html.Div([
        html.Div([
            dcc.Graph(
                        id='violin_plot',
                        figure=make_violin_plot()),
                ], className='mini_container',
                   id='violin',
                ),
        html.Div([
            dcc.Dropdown(id='length_width_dropdown',
                        options=[{'label': 'Thickness', 'value': 'Thickness Material A'},
                                 {'label': 'Width', 'value': 'Width Material Attri'}],
                        value=['Width Material Attri'],
                        multi=True,
                        placeholder="Include in sunburst chart...",
                        className="dcc_control"),
            dcc.Graph(
                        id='sunburst_plot',
                        figure=make_sunburst_plot()),
                ], className='mini_container',
                   id='sunburst',
                ),
            ], className='row container-display',
            ),
    html.Div([
        html.P('Overlay Violin Data:'),
        daq.BooleanSwitch(
          id='daq-violin',
          on=True,
          style={"margin-bottom": "10px", "margin-left": "0px",
          'display': 'inline-block'}),
        dcc.Graph(
                    id='ebit_plot',
                    figure=make_ebit_plot(production_df)),
            ], className='mini_container',
            ),
    html.H4(["Asset Capability"]),
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
                    id='opportunity',
                ),
            ], className='row container-display',
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
    html.H4("Rate, Yield, & Uptime"),
    html.Div([
        html.Div([
            html.Div([
                html.P('Line'),
                dcc.Dropdown(id='line-select',
                             options=[{'label': i, 'value': i} for i in \
                                        lines],
                            value='K40',),
                     ],  className='mini_container',
                         id='line-box',
                     ),
            html.Div([
                html.P('Pareto'),
                dcc.Dropdown(id='pareto-select',
                             options=[{'label': 'Thickness', 'value': 'Thickness Material A'},
                                     {'label': 'Product', 'value': 'Product'}],
                            value='Product',),
                    ],className='mini_container',
                      id='pareto-box',
                    ),
            html.Div([
                html.P('Marginal'),
                dcc.Dropdown(id='marginal-select',
                             options=[{'label': 'Rug', 'value': 'rug'},
                                     {'label': 'Box', 'value': 'box'},
                                     {'label': 'Violin', 'value': 'violin'},
                                    {'label': 'Histogram', 'value': 'histogram'}],
                            value='rug',
                             style={'width': '120px'}),
                    ],className='mini_container',
                      id='marginal-box',
                    ),
            ], className='row container-display',
            ),
        ],
        ),
    html.Div([
        dcc.Graph(
                    id='metric-plot',
                    figure=make_metric_plot()),
            ], className='mini_container',
                id='metric',
            ),
    html.Div([
        dcc.Graph(
                    id='utilization_plot',
                    figure=make_utilization_plot()),
            ], className='mini_container',
                id='util',
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
        ],
        ),
        html.Div([
            html.Div([
                dcc.Graph(
                            id='consolidate_plot',
                            figure=make_consolidate_plot()),
                    ], className='mini_container',
                        id='consolidate-box',
                    ),
            html.Div([
                dcc.Graph(
                            id='product-sunburst',
                            figure=make_product_sunburst()),
                    ], className='mini_container',
                        id='product-box',
                    ),
                ], className='row container-display',
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
    Output('sunburst_plot', 'figure'),
    [Input('violin_plot', 'clickData'),
     Input('length_width_dropdown', 'value'),
     Input('sort', 'value'),
     Input('select', 'value'),
     Input('descriptor_dropdown', 'value')])
def display_sunburst_plot(clickData, toAdd, sort, select, descriptors):
    if sort == 'Best':
        local_df = stat_df.sort_values('score', ascending=False)
        local_df = local_df.reset_index(drop=True)
    else:
        local_df = stat_df
    if descriptors != None:
        local_df = local_df.loc[local_df['descriptor'].isin(descriptors)]
    col = local_df['descriptor'][select[0]]
    val = local_df['group'][select[0]]
    return make_sunburst_plot(clickData, toAdd, col, val)

@app.callback(
    [Output('select', 'max'),
    Output('select', 'value')],
    [Input('descriptor_dropdown', 'value')]
)
def update_descriptor_choices(descriptors):
    max_value = stat_df.loc[stat_df['descriptor'].isin(descriptors)].shape[0]
    value = min(10, max_value)
    return max_value, [0, value]

@app.callback(
    Output('descriptor-number', 'children'),
    [Input('select', 'value')]
)
def display_descriptor_number(select):
    return "Number of Descriptors: {}".format(select[1]-select[0])

@app.callback(
    Output('violin_plot', 'figure'),
    [Input('sort', 'value'),
    Input('select', 'value'),
    Input('descriptor_dropdown', 'value')]
)
def display_violin_plot(sort, select, descriptors):
    return make_violin_plot(sort, select, descriptors)

@app.callback(
    Output('ebit_plot', 'figure'),
    [Input('sort', 'value'),
    Input('select', 'value'),
    Input('descriptor_dropdown', 'value'),
    Input('daq-violin', 'on')]
)
def display_ebit_plot(sort, select, descriptors, switch):
    if switch == True:
        select = list(np.arange(select[0],select[1]))
        return make_ebit_plot(production_df, select, sort=sort, descriptors=descriptors)
    else:
        return make_ebit_plot(production_df)

@app.callback(
    [Output('margin-new-rev', 'children'),
     Output('margin-new-rev-percent', 'children'),
     Output('margin-new-products', 'children')],
    [Input('sort', 'value'),
    Input('select', 'value'),
    Input('descriptor_dropdown', 'value')]
)
def display_opportunity(sort, select, descriptors):
    return calculate_margin_opportunity(sort, select, descriptors)

@app.callback(
    Output('metric-plot', 'figure'),
    [Input('line-select', 'value'),
    Input('pareto-select', 'value'),
    Input('marginal-select', 'value')]
)
def display_opportunity(line, pareto, marginal):
    return make_metric_plot(line, pareto, marginal)

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
    Output('product-sunburst', 'figure'),
    [Input('line-in-selection', 'value'),
     Input('line-out-selection', 'value')]
     )
def display_click_data(inline, outline):
    lines = [inline, outline]
    return make_product_sunburst(lines)

# @app.callback(
#     Output('quantile-target', 'children'),
#     [Input('line-in-selection', 'value'),
#      Input('line-out-selection', 'value'),
#      Input('daq-switch', 'on'),
#      Input('uptime-slider', 'value')]
#      )
# def display_click_data(inline, outline, switch, uptime):
#     if switch == True:
#         quantile, final = find_quantile(inline, outline, ['Rate', 'Yield'],
#                     uptime)
#         return "Quantile-Performance Target: {} + {} Uptime Days"\
#             .format(quantile, uptime)
#     else:
#         quantile, final = find_quantile(inline, outline)
#         return "Quantile-Performance Target: {}".format(quantile)

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
