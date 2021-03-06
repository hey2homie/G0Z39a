import os

import pandas as pd
import json

import flask
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
from actual_code.models import RidgeLassoBuilder, TreeModelBuilder
from skimage import io

# ridge_model = RidgeLassoBuilder("data/final_data/final_data.csv", 0)
# ridge_plots = [ridge_model.get_plot_alpha(), ridge_model.get_plot_cv()]
# lasso_model = RidgeLassoBuilder("data/final_data/final_data.csv", 1)
# lasso_model = [lasso_model.get_plot_alpha(), lasso_model.get_plot_cv()]
# random_forest_model = TreeModelBuilder("data/final_data/final_data.csv")
# random_forest_model.save_plot(0)
# random_forest_model.save_plot(5)
# importance = random_forest_model.important_features()

df_main = [pd.read_excel("./data/prepared_indexes/2013.xlsx"), pd.read_excel("./data/prepared_indexes/2016.xlsx"),
           pd.read_excel("./data/prepared_indexes/2020.xlsx")]
features = pd.read_csv("data/final_data/final_data.csv")
individual_indices_2016 = [pd.read_excel("./data/prepared_indexes/all_kd//2016/" + x) for x in
                           list(os.listdir("./data/prepared_indexes/all_kd/2016/")[::-1])]
individual_indices_2020 = [pd.read_excel("./data/prepared_indexes/all_kd/2020/" + x) for x in
                           list(os.listdir("./data/prepared_indexes/all_kd/2020/")[::-1])]
interesting_columns = ["Rural population", "Urban population", "GDP per capita",
                       "Total population with access to safe drinking-water", "Total water withdrawal",
                       "Water Stress", "Human Development Index", "Water Use Efficiency"]
countries_by_region = [
    ["Afghanistan", "Armenia", "Azerbaijan", "Georgia", "Kazakhstan", "Kyrgyzstan", "Pakistan", "Tajikistan",
     "Turkmenistan", "Uzbekistan"],
    ["Mongolia", "China", "Taiwan"],
    ["Cook Islands", "Fiji", "Kiribati", "Marshall Islands", "Micronesia", "Nauru", "Palau", "Papua New Guinea",
     "Samoa", "Solomon Islands", "Timor-Leste", "Tonga", "Tuvalu", "Vanuatu"],
    ["Bangladesh", "Bhutan", "India", "Maldives", "Nepal", "Sri Lanka"],
    ["Cambodia", "Indonesia", "Lao  PDR", "Malaysia", "Myanmar", "Philippines", "Thailand", "Vietnam"],
    ["Australia", "Brunei Darussalam", "Hong Kong, China", "Japan", "New Zealand", "Republic of Korea", "Singapore"],
    ["Central and West Asia", "East Asia", "Pacific Asia", "South Asia", "South-East Asia", "Advanced Economies"]
]
models_accuracy = {
    "Tuned Random Forest": 0.82, "Boost": 0.72, "AdaBoost": 0.82, "Bagging": 0.77,
    "SVM": 0.59, "Tuned SVM": 0.86,
    "Ridge Regression": 0.86, "Lasso Regression": 0.64
}
importance = [[0.01197942, 0.0252544, 0.01870545, 0.02148561, 0.05945173, 0.05176992, 0.06291193, 0.02242445,
               0.05262889, 0.0486974, 0.01721376, 0.01470227, 0.02198866, 0.04941561, 0.0157635, 0.06251182, 0.01953665,
               0.01750288, 0.03589995, 0.0276397, 0.02569802, 0.01489663, 0.00777421, 0.01349954, 0.01655057,
               0.06827465, 0.01630206, 0.01966815, 0.03211571, 0.02053284, 0.07930952, 0.01780274, 0.01009135],
              ["Total area of the country", "Rural population", "Urban population",
               "Population density", "GDP per capita", "Agriculture, value added",
               "Urban population with access to safe drinking-water",
               "Total renewable water resources per capita", "Total population with access to safe drinking-water",
               "Rural population with access to safe drinking-water",
               "Total water withdrawal per capita", "Total water withdrawal",
               "Water Stress", "Agricultural water withdrawal as % of total water withdrawal",
               "Agricultural water withdrawal", "Human Development Index",
               "Municipal water withdrawal", "Industrial water withdrawal",
               "Industrial Water Use Efficiency", "Irrigated Agriculture Water Use Efficiency", "Water Use Efficiency",
               "Dam capacity per capita", "Resilience and securityStorage drought duration index",
               "Nutrient security", "National Rainfall Index", "Mortality rate, infant", "longitude", "latitude",
               "population ages 0-14", "Population ages 65 and above",
               "Net official development assistance and official aid received", "% of total cultivated area drained",
               "Total exploitable water resources"]]
allindex = pd.read_csv("./data/final_data/allindex2020.csv")
preidctions_future = pd.read_csv("./data/final_data/allindex2025.csv")

# I didn't want to make this mess, adding images to Plotly from files and making it look okay is quite challenging
figure1 = px.imshow(io.imread("data/figs/random_forest_0.png"), width=800, height=900)
figure1.update_layout(width=900, height=1000, title="Tree of estimator 0", title_x=0.5)
figure1.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
figure2 = px.imshow(io.imread("data/figs/random_forest_5.png"), width=800, height=900)
figure2.update_layout(width=900, height=1000, title="Tree of estimator 5", title_x=0.5)
figure2.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)


def get_figure_regressions(number, type):
    fig = px.imshow(io.imread("data/figs/" + type + str(number) + ".png"), width=700, height=700)
    fig.update_layout(width=700, height=700, title_x=0.5)
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    return fig


def get_heatmap(number):
    if number == 1:
        dataframe = allindex
        title = "Heatmap of the Predicted National Water Security Index in 2020"
    else:
        dataframe = preidctions_future
        title = "Heatmap of the Predicted National Water Security Index for 2025 in Selected Countries"
    return px.choropleth_mapbox(dataframe,
                                geojson=countries,
                                locations="Country",
                                color="Water Security Index",
                                color_continuous_scale="sunsetdark",
                                range_color=(1, 5),
                                mapbox_style="carto-positron",
                                zoom=2,
                                center={"lat": 23.4037, "lon": 87.1952},
                                title=title
                                ).update(layout=dict(title=dict(x=0.5)))

with open("actual_code/custom.geo.json", "r", encoding="utf-8") as f:
    countries = json.load(f)

for i in countries["features"]:
    i["id"] = (i["properties"]["name_long"])

server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], server=server)

app.layout = html.Div(
    className="Main",
    children=[
        html.H1(
            "Asia Water Development Outlooks",
            style={
                "text-align": "center"
            }
        ),
        html.H2(
            "Data Exploration",
            style={
                "text-align": "center",
                "margin-top": 40,
                "margin-bottom": 25
            }
        ),
        html.Div(
            children=[
                dbc.Row(
                    children=[
                        dbc.Col(
                            children=[
                                html.Div(
                                    children=[
                                        dcc.Graph(
                                            id="map",
                                            style={
                                                "margin": "0"
                                            }
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
                                    style={
                                        "width": "50%",
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
                                                2: "2020"
                                            }
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
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
                                        style={
                                            "width": "80%",
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
                            )
                        ),
                        dbc.Col(
                            html.Div(
                                children=[
                                    dcc.Graph(
                                        id="radar2",
                                    ),
                                    html.Div(
                                        style={
                                            "width": "80%",
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
                                            )
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
                                            id="barchart",
                                            style={
                                                "margin-top": 50,
                                            }
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
                                    style={
                                        "width": "80%",
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
                                    style={
                                        "width": "80%",
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
                    ],
                    style={
                        "margin-bottom": 50
                    }
                )
            ]
        ),
        html.Div(
            children=[
                dcc.Graph(
                    id="features"
                ),
                html.Div(
                    style={
                        "width": "50%",
                        "margin-left": "25%"
                    },
                    children=[
                        dcc.Slider(
                            id="features_slider",
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
        html.H2(
            "Data Analysis",
            style={
                "text-align": "center",
                "margin-top": 40,
                "margin-bottom": 25,
            }
        ),
        html.Div(
            children=[
                dcc.Graph(
                    id="models",
                    figure=go.Figure(
                        [
                            go.Bar(
                                x=list(models_accuracy.keys()),
                                y=list(models_accuracy.values()),
                                marker_color="#fa826b"
                            )
                        ],
                        layout=go.Layout(
                            title=go.layout.Title(
                                text="Model Performance"
                            ),
                            template="none",
                            yaxis_range=[0, 1]
                        )
                    )
                ),
                html.H3(
                    "Random Forest",
                    style={
                        "text-align": "center",
                        "margin-top": 40,
                        "margin-bottom": 25,
                    }
                ),
                html.Div(
                    children=[
                        dcc.Graph(
                            figure=go.Figure(
                                data=[
                                    go.Bar(
                                        x=importance[0],
                                        y=importance[1],
                                        marker=dict(color=importance[0], colorscale="sunsetdark"),
                                        orientation="h"
                                    )
                                ],
                                layout=go.Layout(
                                    title=go.layout.Title(
                                        text="Important Features in Random Forest Model"
                                    ),
                                    template="none",
                                    yaxis={"visible": False, "showticklabels": False},
                                )
                            )
                        ),
                        dcc.Graph(
                          figure=get_heatmap(1)
                        ),
                        dcc.Graph(
                          figure=get_heatmap(2)
                        ),
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    children=[
                                        dcc.Graph(
                                            figure=figure1
                                        )
                                    ]
                                ),
                                dbc.Col(
                                    children=[
                                        dcc.Graph(
                                            figure=figure2
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                ),
                html.H3(
                    "Ridge Regression",
                    style={
                        "text-align": "center",
                        "margin-top": 40,
                        "margin-bottom": 25,
                    }
                ),
                dbc.Row(
                    children=[
                        dbc.Col(
                            children=[
                                dcc.Graph(
                                    figure=get_figure_regressions(0, "Coef_")

                                )
                            ]
                        ),
                        dbc.Col(
                            children=[
                                dcc.Graph(
                                    figure=get_figure_regressions(0, "Alpha_")
                                )
                            ]
                        )
                    ]
                ),
                html.H3(
                    "Lasso Regression",
                    style={
                        "text-align": "center",
                        "margin-top": 40,
                        "margin-bottom": 25,
                    }
                ),
                dbc.Row(
                    children=[
                        dbc.Col(
                            children=[
                                dcc.Graph(
                                    figure=get_figure_regressions(1, "Coef_")
                                )
                            ]
                        ),
                        dbc.Col(
                            children=[
                                dcc.Graph(
                                    figure=get_figure_regressions(1, "Alpha_")
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
    Output("map", "figure"),
    Input("map_slider", "value"))
def update_map(value):
    df = df_main[value]
    fig = px.choropleth_mapbox(df,
                               geojson=countries,
                               locations="Country",
                               color="National Water Security Index",
                               color_continuous_scale="sunsetdark",
                               range_color=(1, 5),
                               mapbox_style="carto-positron",
                               zoom=2,
                               center={"lat": 23.4037, "lon": 87.1952})
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
    [Input("radar_slider1", "value"),
     Input("radar_slider2", "value")])
def update_radars(region1, region2):
    def create_figure(value, key=None, left=False, visible=True):
        if value == 2016:
            if left:
                data = df_main[1]
            else:
                data = individual_indices_2016[key]
            line_color = "#fed895"
        else:
            if left:
                data = df_main[2]
            else:
                data = individual_indices_2020[key]
            line_color = "#771963"
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
                               visible=visible
                               )

    fig_left = go.Figure(
        data=[
            create_figure(2020, left=True),
            create_figure(2016, left=True)
        ],
        layout=go.Layout(
            title=go.layout.Title(text="National Water Security Scores in " + str(countries_by_region[6][region1])),
            template="none",
            polar=dict(
                radialaxis=dict(range=[0, 20], showticklabels=True, ticks=""),
                angularaxis=dict(showticklabels=True, ticks="")
            ),
            margin=dict(
                b=30
            )
        )
    )

    fig_right = go.Figure(
        layout=go.Layout(
            title="Insights into Key Dimensions",
            template="none",
            polar=dict(
                radialaxis=dict(range=[0, 5], showticklabels=True, ticks=""),
                angularaxis=dict(showticklabels=True, ticks="")
            )
        )
    )

    visible = True
    for dim in range(0, 5):
        fig_right.add_trace(
            create_figure(2016, dim, left=False, visible=visible),
        )
        fig_right.add_trace(
            create_figure(2020, dim, left=False, visible=visible),
        )
        visible = False

    fig_right.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active=0,
            showactive=True,
            buttons=list([
                dict(label="K1",
                     method="update",
                     args=[{"visible": [True, True, False, False, False, False, False, False, False, False]},
                           {"showlegend": True}]),
                dict(label="K2",
                     method="update",
                     args=[{"visible": [False, False, True, True, False, False, False, False, False, False]},
                           {"showlegend": True}]),
                dict(label="K3",
                     method="update",
                     args=[{"visible": [False, False, False, False, True, True, False, False, False, False]},
                           {"showlegend": True}]),
                dict(label="K4",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, True, True, False, False]},
                           {"showlegend": True}]),
                dict(label="K5",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, False, False, True, True]},
                           {"showlegend": True}]),
            ])
        )],
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
def update_key_dimension_results(year, region):
    def create_figure(kd):
        colors = [" #fed895", " #fa826b", " #e54565", " #c9266d", " #771963"]
        df = df_main[year]
        areas = df[df["Country"].isin(countries_by_region[region])].iloc[:, 0]
        return go.Bar(
            name="KD" + str(kd),
            x=areas,
            y=df.iloc[df[df["Country"].isin(countries_by_region[region])].index.values, kd].values,
            marker_color=(colors[kd - 1] * len(areas)).split(" ")[1:]
        )

    if year == 0:
        scale = [0, 20]
    else:
        scale = [0, 100]

    fig = go.Figure(
        data=[
            create_figure(x) for x in range(1, 6)
        ],
        layout=go.Layout(
            template="none"
        )
    )
    fig.update_layout(title_text="Bar Chart of Overall Index Score in a Particular Region",
                      title_x=0.5,
                      margin=dict(
                          b=30
                      ),
                      barmode="stack",
                      yaxis_range=scale
                      )
    return fig


@app.callback(
    Output("features", "figure"),
    Input("features_slider", "value")
)
def update_features(region):
    def create_figure(column, visible):
        df = features
        areas = df[df["country"].isin(countries_by_region[region])].iloc[:, 0]
        return go.Bar(
            x=areas,
            y=df.iloc[df[df["country"].isin(countries_by_region[region])].index.values,
                      df.columns.get_loc(column)].values,
            marker_color="#fed895",
            visible=visible,
        )

    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text="Insight into 2020 data set columns by region (data is normalized)"),
            template="none"
        )
    )

    visible = True
    for i in interesting_columns:
        fig.add_trace(
            create_figure(i, visible)
        )
        visible = False

    fig.update_layout(
        updatemenus=[
            go.layout.Updatemenu(
                active=0,
                showactive=True,
                buttons=list([
                    dict(label="Rural population",
                         method="update",
                         args=[{"visible": [True, False, False, False, False, False, False, False]}]),
                    dict(label="Urban population",
                         method="update",
                         args=[{"visible": [False, True, False, False, False, False, False, False]}]),
                    dict(label="GDP per capita",
                         method="update",
                         args=[{"visible": [False, False, True, False, False, False, False, False]}]),
                    dict(label="Total population with access to safe drinking-water",
                         method="update",
                         args=[{"visible": [False, False, False, True, False, False, False, False]}]),
                    dict(label="Total water withdrawal",
                         method="update",
                         args=[{"visible": [False, False, False, False, True, False, False, False]}]),
                    dict(label="Water Stress",
                         method="update",
                         args=[{"visible": [False, False, False, False, False, True, False, False]}]),
                    dict(label="Human Development Index",
                         method="update",
                         args=[{"visible": [False, False, False, False, False, False, True, False]}]),
                    dict(label="Water Use Efficiency",
                         method="update",
                         args=[{"visible": [False, False, False, False, False, False, False, True]}]),
                ]),
                x=0.5,
                xanchor="center",
                y=0.9,
                yanchor="bottom",
                pad={"t": 40, "b": 40}
            )
        ],
        margin=dict(
            b=30
        )
    )

    return fig


if __name__ == "__main__":
    app.run_server()
