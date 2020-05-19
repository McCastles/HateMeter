import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
from dash.dependencies import Input, Output
import DBConnector as database


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H4('Live Twitter Data from MongoDB (1sec intervals)'),
        html.Div(id='live-update-text'),
        dcc.Interval(
            id='interval-component',
            interval=database.DELAY_SEC*1000,
            n_intervals=0
        )
    ])
)


@app.callback(
    Output('live-update-text', 'children'),
    [Input('interval-component', 'n_intervals')])
def update_data_display(n):
    # print(n_intervals)
    doc = database.get_next_post('DHL')
    col_names = ['created', 'score', 'sentiment', 'text', 'username']
    style = {'padding': '5px', 'fontSize': '16px'}
    return [
        html.Span('created: {}'.format(doc['created']), style=style),
        html.Span('score: {0:.2f}'.format(doc['score']), style=style),
        html.Span('sentiment: {}'.format(doc['sentiment']), style=style),
        html.Span('text: {}'.format(doc['text']), style=style),
        html.Span('username: {}'.format(doc['username']), style=style)
        # html.Span('{}: {0:.2f}'.format(
            # col_name, doc[col_name], style=style))
            # for col_name in col_names      
    ]




if __name__=='__main__':

    host = '0.0.0.0'
    app.run_server(debug=True, host=host)