import chart_studio
import chart_studio.plotly as py
import pandas as pd
import plotly.express as px
from api_info.plotly_api import USERNAME, API_KEY

def main():
    chart_studio.tools.set_credentials_file(username=USERNAME, api_key=API_KEY)
    graph_data = pd.read_csv("forest40-plotly.csv")
    fig = px.scatter(graph_data,
                        x="predicted",
                        y="revenue",
                        hover_name="title",
                        hover_data=["predicted", "revenue", "difference", "percent_off"],
                        title=f"Actual Revenue and Predicted Revenue",
                        width=800,
                        height=800)
    fig.update_traces(marker=dict(size=12,
                                color='skyblue',
                                line=dict(width=1,
                                            color='black')))
    fig.update_layout(hovermode="closest",
                    hoverlabel=dict(
                            bgcolor="skyblue",
                            font_size=16,
                            font_family="Rockwell"))
    py.plot(fig, filename="Random Forest 40 Trees", auto_open=True)

if __name__ == "__main__":
    main()