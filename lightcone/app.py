import torch
from dash import (
    Dash,
    dcc,
    html,
    Input,
    Output,
)
from jupyter_dash import JupyterDash
import plotly.express as px
import plotly.graph_objects as go


def make_app(model, embedding):

    app = JupyterDash('ligthcone')

    figure = go.Figure(
        data=[
            go.Scattergl(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode='markers',
                name='data',
            ),
        ],
    )

    figure.update_layout(
        autosize=False,
        width=500,
        height=500,
        showlegend=False,
    )

    app.layout = html.Div([
        html.Div(
            [
                html.Div(
                    dcc.Graph(
                        id="latent_space",
                        figure=figure
                    ),
                    style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}

                ),
                html.Div(
                    id='reconstruction',
                    style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}
                ),
            ],
            style={
                'width': '100%',
                'display': 'inline-block',
            }
        )
    ])

    @app.callback(Output('reconstruction', 'children'), Input('latent_space', 'clickData'))
    def decode(data):

        if data is None:
            return []

        x = torch.Tensor([
            [
                data['points'][0]['x'],
                data['points'][0]['y'],
            ]
        ])

        fig = px.imshow(
            model.decoder(x).cpu().detach().numpy()[0][0],
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        return dcc.Graph(
            id="recon",
            figure=fig,
        )

    return app


def run(model, embedding=None, port='6007', host='0.0.0.0'):
    """
    """

    app = make_app(model=model, embedding=embedding)
    app.run_server(
        debug=True,
        port=port,
        mode='inline',
    )
