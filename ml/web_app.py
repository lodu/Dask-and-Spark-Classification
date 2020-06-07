import dash
import dash_core_components as dcc
import dash_html_components as html

import yaml


try:
    with open(r"spark_stats.yaml") as file:
        spark_data = yaml.load(file, Loader=yaml.FullLoader)
except FileNotFoundError:
    print("Run the spark modeller first.")
    exit(0)

try:
    with open(r"dask_stats.yaml") as file:
        dask_data = yaml.load(file, Loader=yaml.FullLoader)
except FileNotFoundError:
    print("Run the dask modeller first.")
    exit(0)


auc_spark = spark_data["auc"] / 1000
speed_spark = spark_data["timing"] / 1000


auc_dask = dask_data["auc"] / 1000
speed_dask = dask_data["timing"] / 1000
app = dash.Dash()
colors = {"background": "#111111", "text": "#7FDBFF"}
app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.H1(
            children="Web overview of performance comparison",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        html.Div(
            children="1 = Dask and 2 = Spark",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        dcc.Graph(
            id="Performance",
            figure={
                "data": [
                    {
                        "x": [1, 2],
                        "y": [auc_dask, auc_spark],
                        "type": "bar",
                        "name": "AUC",
                    },
                    {
                        "x": [1, 2],
                        "y": [speed_dask, speed_spark],
                        "type": "bar",
                        "name": "Speed (s)",
                    },
                ],
                "layout": {
                    "plot_bgcolor": colors["background"],
                    "paper_bgcolor": colors["background"],
                    "font": {"color": colors["text"]},
                },
            },
        ),
    ],
)

if __name__ == "__main__":
    app.run_server(debug=True)
