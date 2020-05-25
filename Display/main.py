import dash
import plotly

# Custom files
import DBConnector as database
from callbacks import register_callbacks
from layout import load_layout


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Set up layout in layout.py
app.layout = load_layout(
    database.HISTORY_PERIOD_SEC,
    database.HISTOGRAM_PERIOD_SEC
    )

# Set up callbacks in callbacks.py
register_callbacks(app, database)






if __name__=='__main__':

    host = '0.0.0.0'
    app.run_server(debug=True, host=host)