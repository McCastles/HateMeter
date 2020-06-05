import dash_html_components as html
import dash_core_components as dcc


def load_layout(history_period_sec, histogram_period_sec):
    return html.Div(
        html.Div([

            html.H4('Live Twitter Data from MongoDB (1sec intervals)'),

            # For history
            html.Div(id='history-live-update-text'),
            dcc.Interval(
                id='history-interval-component',
                interval=1000*history_period_sec,
                n_intervals=0
            ),

            # For histogram
            html.Div(id='histogram-live-update-text'),
            dcc.Interval(
                id='histogram-interval-component',
                interval=1000*histogram_period_sec,
                n_intervals=0
            ),

            #test graph
            dcc.Graph(
                id='histogram-graph'
            )

        ])
    )

