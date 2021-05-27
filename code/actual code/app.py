import os

import pandas as pd

import json
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import plotly.express as px
import plotly.graph_objs as go

# Dataframes and lists
df_main = [pd.read_excel("../../data/prepared_indexes/2013.xlsx"),
           pd.read_excel("../../data/prepared_indexes/2016.xlsx"),
           pd.read_excel("../../data/prepared_indexes/2020.xlsx")]

individual_indices_2016 = [pd.read_excel("../../data/prepared_indexes/all_kd//2016/" + x) for x in
                           list(os.listdir("../../data/prepared_indexes/all_kd/2016/")[::-1])]
individual_indices_2020 = [pd.read_excel("../../data/prepared_indexes/all_kd/2020/" + x) for x in
                           list(os.listdir("../../data/prepared_indexes/all_kd/2020/")[::-1])]

countries_by_region = [
    ["Afghanistan", "Armenia", "Azerbaijan", "Georgia", "Kazakhstan", "Kyrgyzstan", "Pakistan", "Tajikistan",
     "Turkmenistan", "Uzbekistan"],
    ["Mongolia", "China", "Taiwan"],
    ["Cook Islands", "Fiji", "Kiribati", "Marshall Islands", "Micronesia", "Nauru", "Palau", "Papua New Guinea",
     "Samoa", "Solomon Islands", "Timor-Leste", "Tonga", "Tuvalu", "Vanuatu"],
    ["Bangladesh", "Bhutan", "India", "Maldives", "Nepal", "Sri Lanka"],
    ["Cambodia", "Indonesia", "Lao  PDR", "Malaysia", "Myanmar", "Philippines", "Thailand", "Vietnam"],
    ["Australia", "Brunei Darussalam", "Hong Kong, China", "Japan", "New Zealand", "Korea", "Singapore"],
    ["Central and West Asia", "East Asia", "Pacific Asia", "South Asia", "South-East Asia", "Advanced Economies"]
]

# JSON
with open("custom.geo-4.json", "r", encoding="utf-8") as f:
    countries = json.load(f)

for i in countries["features"]:
    i["id"] = (i["properties"]["name"])

# App
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    className="Main",
    children=[
        html.H1("Exploring the Asia Water Development Outlooks",
                style={
                    "left-margin": "10%",
                    "right-margin": "10%",
                    'text-align':'center'
                }),
        html.Div(
            children=[
                dbc.Row(
                    children=[
                        dbc.Col(
                            children=[
                                html.Div(
                                    className="Map",
                                    children=[
                                        dcc.Graph(
                                            id="map",
                                            style={
                                                "margin": "0"
                                            },
                                        ),
                                    ]
                                )
                            ]
                        )
                    ]
                ),
                dbc.Row(
                    children=[
                        dbc.Col(
                            children=[
                                html.Div(
                                    className="SliderMap",
                                    style={
                                        'width': '50%',
                                        "margin-left": "25%"
                                    },
                                    children=[
                                        dcc.Slider(
                                            id="map_slider",
                                            min=0,
                                            max=2,
                                            step=None,
                                            value=0,
                                            marks={
                                                0: "2013",
                                                1: "2016",
                                                2: "2020",
                                            }
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                )
            ]
        ),
        html.Div(
            children=[
                dbc.Row(
                    children=[
                        dbc.Col(
                            html.Div(
                                children=[
                                    dcc.Graph(
                                        id="radar1",
                                    ),
                                    html.Div(
                                        className="Radar Chart",
                                        style={
                                            'width': '80%',
                                            "margin-left": "10%",
                                            "margin-right": "10%"
                                        },
                                        children=[
                                            dcc.Slider(
                                                id="radar_slider1",
                                                min=0,
                                                max=5,
                                                step=None,
                                                value=0,
                                                marks={
                                                    0: "Central and West Asia",
                                                    1: "East Asia",
                                                    2: "Pacific Asia",
                                                    3: "South Asia",
                                                    4: "South-East Asia",
                                                    5: "Advanced Economies"
                                                }
                                            )
                                        ]
                                    )
                                ]
                            ),
                        ),
                        dbc.Col(
                            html.Div(
                                children=[
                                    dcc.Graph(
                                        id="radar2",
                                    ),
                                    html.Div(
                                        className="Radar Chart",
                                        style={
                                            'width': '80%',
                                            "margin-left": "10%",
                                            "margin-right": "10%"
                                        },
                                        children=[
                                            dcc.Slider(
                                                id="radar_slider2",
                                                min=0,
                                                max=5,
                                                step=None,
                                                value=0,
                                                marks={
                                                    0: "Central and West Asia",
                                                    1: "East Asia",
                                                    2: "Pacific Asia",
                                                    3: "South Asia",
                                                    4: "South-East Asia",
                                                    5: "Advanced Economies"
                                                }
                                            ),
                                        ]
                                    )
                                ]
                            )
                        )
                    ]
                ),
                dbc.Row(
                    children=[
                        dbc.Col(
                            children=[
                                html.Div(
                                    children=[
                                        dcc.Graph(
                                            id="barchart"
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                ),
                dbc.Row(
                    children=[
                        dbc.Col(
                            children=[
                                html.Div(
                                    className="BarChart_slider",
                                    style={
                                        'width': '80%',
                                        "margin-left": "10%",
                                        "margin-right": "10%"
                                    },
                                    children=[
                                        dcc.Slider(
                                            id="barchart_slider_1",
                                            min=0,
                                            max=2,
                                            step=None,
                                            value=0,
                                            marks={
                                                0: "2013",
                                                1: "2016",
                                                2: "2020"
                                            }
                                        )
                                    ]
                                )
                            ]
                        ),
                        dbc.Col(
                            children=[
                                html.Div(
                                    className="BarChart_slider",
                                    style={
                                        'width': '80%',
                                        "margin-left": "10%",
                                        "margin-right": "10%"
                                    },
                                    children=[
                                        dcc.Slider(
                                            id="barchart_slider_2",
                                            min=0,
                                            max=5,
                                            step=None,
                                            value=0,
                                            marks={
                                                0: "Central and West Asia",
                                                1: "East Asia",
                                                2: "Pacific Asia",
                                                3: "South Asia",
                                                4: "South-East Asia",
                                                5: "Advanced Economies"
                                            }
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)


@app.callback(
    Output('map', 'figure'),
    Input('map_slider', 'value'))
def update_map(value):
    df = df_main[value]
    fig = px.choropleth_mapbox(df,
                               geojson=countries,
                               locations='Country',
                               color='National Water Security Index',
                               color_continuous_scale="sunsetdark",
                               range_color=(1, 5),
                               mapbox_style="carto-positron",
                               zoom=2,
                               center={'lat': 23.4037, "lon": 87.1952})
    fig.update_layout(title_text="Heatmap of National Water Security Index by Year",
                      title_x=0.5,
                      margin=dict(
                          b=10
                        )
                      )
    return fig


@app.callback(
    [Output("radar1", "figure"),
     Output("radar2", "figure")],
    [Input('radar_slider1', 'value'),
     Input('radar_slider2', 'value')])
def show_radar(region1, region2):
    def create_figure(value, key=None, left=False):
        if value == 2016:
            if left:
                data = df_main[1]
            else:
                data = individual_indices_2016[key]   # TODO: make a parameter
            line_color = "#fed895"
        else:
            if left:
                data = df_main[2]
            else:
                data = individual_indices_2020[key]   # TODO: make a parameter
            line_color = '#771963'
        if left:
            sliced = data[data["Country"].isin(countries_by_region[region1])].iloc[:, 1:6]
            theta = data.columns[1:6]
        else:
            sliced = data[data["Country"].isin(countries_by_region[region2])].iloc[:, 1:- 1]
            theta = data.columns[1:-1]

        sliced = [round(num, 0) for num in sliced.mean(axis=0).values.tolist()]
        theta = list(theta)
        sliced.append(sliced[0])
        theta.append(theta[0])

        return go.Scatterpolar(r=sliced,
                               theta=theta,
                               fill="toself",
                               fillcolor=line_color,
                               opacity=0.5,
                               name="Year " + str(value),
                               )

    fig_left = go.Figure(
        data=[
            create_figure(2020, left=True),
            create_figure(2016, left=True)
        ],
        layout=go.Layout(
            title=go.layout.Title(text='National Water Security Scores in ' + str(countries_by_region[6][region1])),
            template="none",
            polar=dict(
                radialaxis=dict(range=[0, 20], showticklabels=True, ticks=''),
                angularaxis=dict(showticklabels=True, ticks='')
            ),
            margin=dict(
                b=30
            )
        )
    )

    fig_right = go.Figure(
        layout=go.Layout(
            title="Insight into Key Dimension",
            template="none",
            polar=dict(
                radialaxis=dict(range=[0, 5], showticklabels=True, ticks=''),
                angularaxis=dict(showticklabels=True, ticks='')
            )
        )
    )

    for dim in range(0, 5):
        fig_right.add_trace(
            create_figure(2016, dim, left=False)
        )
        fig_right.add_trace(
            create_figure(2020, dim, left=False)
        )

    fig_right.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active=0,
            showactive=True,
            buttons=list([
                dict(label='All',
                     method='restyle',
                     args=[{'visible': [True, True, True, True, True, True, True, True, True, True]},
                           {'title': 'Key Dimension 1',
                            'showlegend': True}]),
                dict(label='K1',
                     method='restyle',
                     args=[{'visible': [True, True, False, False, False, False, False, False, False, False]},
                            {'title': 'Key Dimension 1',
                             'showlegend': True}]),
                dict(label='K2',
                     method='restyle',
                     args=[{'visible': [False, False, True, True, False, False, False, False, False, False]},
                            {'title': 'Key Dimension 2',
                             'showlegend': True}]),
                dict(label='K3',
                     method='restyle',
                     args=[{'visible': [False, False, False, False, True, True, False, False, False, False]},
                            {'title': 'Key Dimension 3',
                             'showlegend': True}]),
                dict(label='K4',
                     method='restyle',
                     args=[{'visible': [False, False, False, False, False, False, True, True, False, False]},
                            {'title': 'Key Dimension 4',
                             'showlegend': True}]),
                dict(label='K5',
                     method='restyle',
                     args=[{'visible': [False, False, False, False, False, False, False, False, True, True]},
                           {'title': 'Key Dimension 5',
                            'showlegend': True}]),
                     ]
                )
            )
        ],
        margin=dict(
            b=30
        )
    )
    return [fig_left, fig_right]


@app.callback(
    Output("barchart", "figure"),
    [Input("barchart_slider_1", "value"),
     Input("barchart_slider_2", "value")]
)
def update_barchart(year, region):
    def create_figure(kd):
        colors = [" #fed895", " #fa826b", " #e54565", " #c9266d", " #771963"]
        df = df_main[year]
        areas = df[df["Country"].isin(countries_by_region[region])].iloc[:, 0]
        return go.Bar(
            name="KD" + str(kd),
            x=areas,
            y=df.iloc[df[df["Country"].isin(countries_by_region[region])].index.values, kd].values,
            marker_color=(colors[kd-1] * len(areas)).split(" ")[1:]
        )
    if year == 0:
        scale = [0, 20]
    else:
        scale = [0, 100]

    fig = go.Figure(
        data=[
            create_figure(1),
            create_figure(2),
            create_figure(3),
            create_figure(4),
            create_figure(5)
        ],
        layout=go.Layout(
            title=go.layout.Title(text='National Water Security Scores in '),
            template="none"
        )
    )
    fig.update_layout(title_text="Heatmap of National Water Security Index by Year",
                      title_x=0.5,
                      margin=dict(
                          b=30
                        ),
                      barmode='stack',
                      yaxis_range=scale
                      )
    return fig


if __name__ == '__main__':
    app.run_server()
