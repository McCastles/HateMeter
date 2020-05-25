
import dash_html_components as html
from dash.dependencies import Input, Output


# Inside this function we define custom callbacks
# Dont worry, it works, just write your regular callbacks inside
def register_callbacks(app, database):



    # This one reads DHL collection from the beginning periodically
    # Period is defined in history-interval-component in layout.py
    
    @app.callback(
        Output('history-live-update-text', 'children'),
        [Input('history-interval-component', 'n_intervals')])
    def update_history_meta(n):

        # TODO: pass comp_name from button.. somehow
        comp_name = 'fedex'
        post = database.update_history(
            comp_name, 
            database.history_counters,
            verbose=False)
        
        col_names = ['CreatedAt', 'Username', 'score', 'sentiment', 'text']
        style = {'padding': '5px', 'fontSize': '16px'}
        
        visual = [
            html.Span(
                f'{col_name}: {post[col_name]}',
                style=style)
            for col_name in col_names
        ]
        
        return visual

    


    @app.callback(
        Output('histogram-live-update-text', 'children'),
        [Input('histogram-interval-component', 'n_intervals')])
    def update_histogram_meta(n):

        # For each company
        # how many tweets were posted
        # within the last HISTOGRAM_PERIOD_SEC seconds
        distribution = database.update_histogram(
            database.histogram_counters
        )
        style = {'padding': '5px', 'fontSize': '16px'}

        visual = [
            html.Span(
                f'{k}: {v}',
                style=style
            ) 
            for k, v in distribution.items()
        ]

        return visual




