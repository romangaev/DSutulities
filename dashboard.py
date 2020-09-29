import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_table
import os
import cx_Oracle
import warnings
import time

warnings.filterwarnings('ignore')

#Set up DB connection
os.environ['ORACLE_HOME'] = "oraclepath"
dns_tns = cx_Oracle.makedsn('ip','7777',service_name = 'servicename')
usr = getpass.getpass(prompt='Insert username:\n')
pwd = getpass.getpass(prompt='Insert password:\n')
conn = cx_Oracle.connect(user=usr, password=pwd, dsn=dns_tns, encoding='utf-8')
cur = conn.cursor()

app = dash.Dash(__name__)

df_cmp = pd.read_sql('''''', con=conn)
df_cmp['START_DT'] = df_cmp['START_DT'].dt.date
df_cmp['END_DT'] = df_cmp['END_DT'].dt.date
df_cmp['COMBINED'] = df_cmp['PRD_NAME'] + '_' + df_cmp['CAMP_ID'].astype(str) + '_' + df_cmp['START_DT'].astype(str)
available_indicators = [{'label': value, 'value': key} for key, value in
                        dict(zip(df_cmp['CAMP_ID'], df_cmp['COMBINED'])).items()]

dyn_all = pd.read_sql('''''', con=conn)
cg_balance_all = pd.read_sql('''''', con=conn)

card_style = {
    'background-color': '#ffffff',
    'margin': '10px',
    'padding': '10px',
    'position': 'relative',
    'box-shadow': '0 0 4px grey'
}

card_style_selected = {
    'background-color': '#ffffff',
    'borderBottom': '10px solid #0341fc',
    'margin': '10px',
    'padding': '10px',
    'position': 'relative',
    'box-shadow': '0 0 4px grey'
}

app.layout = html.Div([
    html.H1('Анализ кампании', style={'margin': '20px'}),
    html.Div([
        html.Div('Кампания для анализа:', style={'flex': '0 1 10%', 'margin': '10px'}),

        html.Div(
            dcc.Dropdown(
                id='campaign',
                options=available_indicators,
                value=1
            ), style={'flex': '0 1 20%', 'margin-top': '10px', 'margin-bottom': '10px'}),

        html.Div(id="info", style={'flex': '0 1 70%', 'margin': '10px', 'margin-left': '50px'})

    ], style={'display': 'flex', 'flex': '0 1 auto', 'flex-flow': 'row nowrap', 'margin': '10px'}),

    html.Div(id='metrics', style={'display': 'flex', 'flex': '0 1 auto', 'flex-flow': 'row nowrap', 'margin': '10px'}),

    html.Div([
        html.Div([
            dcc.Graph(id='fun')
        ], style={'flex': '0 1 30%', 'margin': '10px'}),
        html.Div([
            dcc.Graph(id='dyn'),
            dcc.RadioItems(id='dyn_radio',options=[{'label':'Динамика целевых событий','value':'TAR_ACT'},
                                                    {'label':'Динамика конверсий','value':'CONV'}],
                                                     value='TAR_ACT')
        ], style={'flex': '0 1 70%', 'margin': '10px'})
    ], style={'display': 'flex', 'flex': '0 1 auto', 'flex-flow': 'row nowrap', 'margin': '10px'}),
    html.H1('Сводная динамика всех кампаний', style={'margin-top': '50px', 'margin-left':'10px'}),
    html.Div([
            html.Div(children=[dcc.RadioItems(id='all_radio',options=[{'label':'Динамика целевых событий','value':'TAR_ACT'},
                                                                        {'label':'Динамика прироста конверсий','value':'CONV_INCR'}],
                                                     value='CONV_INCR',labelStyle={'display':'block','margin-bottom':'10px'})], style ={'margin-right':'50px'}),
            daq.BooleanSwitch(id='date_align_switch',on=True, color='#0341fc', label='Привести к одной дате')
        ], style={'display': 'flex', 'flex': '0 1 auto', 'flex-flow': 'row nowrap', 'margin': '10px','vertical-align':'middle'}),
    dcc.Graph(id='fig_combined')


], style={'font-family': ["Verdana", "Arial", "sans-serif"]})


def create_metrics(campaign):
    df_cmp = pd.read_sql('''select/*+ parallel(32)*/  * from cmp.campaigns''', con=conn)
    cmp_info = \
        df_cmp[df_cmp['CAMP_ID'] == campaign][['DESCRIPTION', 'PRD_NAME', 'TARGET_METRIC', 'MODEL_TYPE', 'START_DT', 'END_DT']].iloc[0]

    df_metrics = pd.read_sql(f'''''', con=conn)

    conv_test = df_metrics[df_metrics['METRIC'] == 'CONV_PERC'][df_metrics['IS_CONTROL'] == 0][
        'METRIC_VALUE'].mean() / 100
    conv_control = \
    df_metrics[df_metrics['METRIC'] == 'CONV_PERC'][df_metrics['IS_CONTROL'] == 1][
        'METRIC_VALUE'].mean() / 100

    if cmp_info['PRD_NAME'] == 'IP_CALLS':
        print('ip')
        metric_2_name = 'Доля IP звонков:'
        metric_2_test = \
        df_metrics[df_metrics['METRIC'] == 'IP_SHARE'][df_metrics['IS_CONTROL'] == 0]['METRIC_VALUE'].mean()
        metric_2_control = \
        df_metrics[df_metrics['METRIC'] == 'IP_SHARE'][df_metrics['IS_CONTROL'] == 1]['METRIC_VALUE'].mean()

        metric_3_name = 'Звонков на клиента:'
        metric_3_test = \
            df_metrics[df_metrics['METRIC'] == 'AVG CALLS PER CLIENT'][df_metrics['IS_CONTROL'] == 0][
                'METRIC_VALUE'].mean()
        metric_3_control = \
            df_metrics[df_metrics['METRIC'] == 'AVG CALLS PER CLIENT'][df_metrics['IS_CONTROL'] == 1][
                'METRIC_VALUE'].mean()

    if cmp_info['PRD_NAME'] == 'BIOM':
        print('biom')
        metric_2_name = 'Доля самоходов в отклике:'
        metric_2_test = \
        df_metrics[df_metrics['METRIC'] == 'SAMOHOD SHARE'][df_metrics['IS_CONTROL'] == 0]['METRIC_VALUE'].mean()
        metric_2_control = \
        df_metrics[df_metrics['METRIC'] == 'SAMOHOD SHARE'][df_metrics['IS_CONTROL'] == 1]['METRIC_VALUE'].mean()

        metric_3_name = '-'
        metric_3_test = \
            df_metrics[df_metrics['METRIC'] == '-'][
                df_metrics['IS_CONTROL'] == 0][
                'METRIC_VALUE'].mean()
        metric_3_control = \
            df_metrics[df_metrics['METRIC'] == '-'][
                df_metrics['IS_CONTROL'] == 1][
                'METRIC_VALUE'].mean()

    metrics = [
        html.Div([
            html.Div([
                html.Div(['Конверсия:']),
                html.Div([
                    html.Div([
                        html.Div('Тест'),
                        html.Div(["{:.2%}".format(conv_test)], style={'font-size': '36px'})
                        ], style={'flex': '0 1 50%', 'margin': '10px'})
                    ,
                    html.Div([
                        html.Div('Контроль'),
                        html.Div(["{:.2%}".format(conv_control)], style={'font-size': '36px'})
                    ], style={'flex': '0 1 50%', 'margin': '10px'})
                ], style={'display': 'flex', 'flex': '0 1 auto', 'flex-flow': 'row nowrap'})

                ], style=card_style_selected),
        ], style={'flex': '0 1 33%'}),

        html.Div([
            html.Div([
                html.Div([metric_2_name]),
                html.Div([
                    html.Div([
                        html.Div('Тест'),
                        html.Div(["{:.2%}".format(metric_2_test)], style={'font-size': '36px'})
                    ], style={'flex': '0 1 50%', 'margin': '10px'})
                    ,
                    html.Div([
                        html.Div('Контроль'),
                        html.Div(["{:.2%}".format(metric_2_control)], style={'font-size': '36px'})
                    ], style={'flex': '0 1 50%', 'margin': '10px'})
                ], style={'display': 'flex', 'flex': '0 1 auto', 'flex-flow': 'row nowrap'})

            ], style=card_style),
        ], style={'flex': '0 1 33%'}),

        html.Div([
            html.Div([
                html.Div([metric_3_name]),
                html.Div([
                    html.Div([
                        html.Div('Тест'),
                        html.Div(["{:.2}".format(metric_3_test)], style={'font-size': '36px'})
                    ], style={'flex': '0 1 50%', 'margin': '10px'})
                    ,
                    html.Div([
                        html.Div('Контроль'),
                        html.Div(["{:.2}".format(metric_3_control)], style={'font-size': '36px'})
                    ], style={'flex': '0 1 50%', 'margin': '10px'})
                ], style={'display': 'flex', 'flex': '0 1 auto', 'flex-flow': 'row nowrap'})

            ], style=card_style),
        ], style={'flex': '0 1 33%'})
    ]
    return metrics


def create_info(campaign):
    df_cmp = pd.read_sql('''''', con=conn)
    df_cmp['START_DT'] = df_cmp['START_DT'].dt.date
    df_cmp['END_DT'] = df_cmp['END_DT'].dt.date
    df_cmp['COMBINED'] = df_cmp['PRD_NAME'] + '_' + df_cmp['CAMP_ID'].astype(str) + '_' + df_cmp['START_DT'].astype(str)

    cmp_info = \
    df_cmp[df_cmp['CAMP_ID'] == campaign][['DESCRIPTION','PRD_NAME', 'TARGET_METRIC', 'MODEL_TYPE', 'START_DT', 'END_DT']].iloc[0]

    passed = (pd.datetime.now().date() - cmp_info['START_DT']).days
    overall = (cmp_info['END_DT'] - cmp_info['START_DT']).days

    cmp_info_html = html.Div([
        html.Div([
            html.Div(['Завершенность:']),
            html.Div(["{:.0%}".format(min([(passed / overall), 1]))], style={'font-size': '36px'})
        ],
            style={'flex': '0 1 20%'}
        ),
        html.Div([
            html.Div(f'''Описание: {cmp_info['DESCRIPTION']}'''),
            html.Br(),
            html.Div(f'''Целевая метрика: {cmp_info['TARGET_METRIC']}'''),
            html.Br(),
            html.Div(f'''Модель: {cmp_info['MODEL_TYPE']}'''),
            html.Br(),
            html.Div(f'''Дата начала: {cmp_info['START_DT']}'''),
            html.Br(),
            html.Div(f'''Дата окончания: {cmp_info['END_DT']}''')
        ], style={'font-size': '10px', 'flex': '0 1 80%'})
    ], style={'display': 'flex', 'flex': '0 1 auto', 'flex-flow': 'row nowrap'}
    )

    return cmp_info_html


def create_dyn(campaign, radio):
    dyn = dyn_all[dyn_all['CAMP_ID']==campaign]
    cg_balance = cg_balance_all[cg_balance_all['CAMP_ID']==campaign]

    cg_imbalance_rate = 1
    if len(cg_balance)>1:
        cg_imbalance_rate = cg_balance[cg_balance['IS_CONTROL']==0]['CNT'].iloc[0]/\
                            cg_balance[cg_balance['IS_CONTROL']==1]['CNT'].iloc[0]

    fig_dyn = go.Figure()

    test_dyn_y = dyn[dyn['IS_CONTROL']==0]['CNT']
    test_dyn_x = dyn[dyn['IS_CONTROL']==0]['TARGET_ACTION_DT']
    control_dyn_y = dyn[dyn['IS_CONTROL']==1]['CNT']*cg_imbalance_rate
    control_dyn_x = dyn[dyn['IS_CONTROL']==1]['TARGET_ACTION_DT']

    conv_test_dyn = dyn[dyn['IS_CONTROL']==0]['CNT'].cumsum()/cg_balance[cg_balance['IS_CONTROL']==0]['CNT'].iloc[0]
    conv_incr_test_dyn = conv_test_dyn.diff()
    
    if len(cg_balance)>1:
        conv_control_dyn = dyn[dyn['IS_CONTROL']==1]['CNT'].cumsum()/cg_balance[cg_balance['IS_CONTROL']==1]['CNT'].iloc[0]
        conv_incr_control_dyn = conv_control_dyn.diff()

    if radio == 'TAR_ACT':
        yt= test_dyn_y
        if len(cg_balance)>1:
            yc = control_dyn_y
        title ='Уникальных клиентов'
    elif radio == 'CONV':
        yt=conv_test_dyn
        if len(cg_balance)>1:
            yc = conv_control_dyn
        title = 'Конверсия'

    fig_dyn.add_trace(go.Bar(x=test_dyn_x,
                                y=yt,
                                name='Тест',
                                marker_color='#0341fc'
                                ))
    if len(cg_balance)>1:
            fig_dyn.add_trace(go.Bar(x=control_dyn_x,
                                    y=yc,
                                    name='Контроль',
                                    marker_color='#ff6536'
            ))
    fig_dyn.update_yaxes(title=title)
    
    fig_dyn.update_layout(margin=dict(l=20, r=20, t=20, b=30), legend=dict(yanchor='top', xanchor='right'))
    
    fig_dyn.update_xaxes(title='Дата', rangeslider_visible=True,
                         rangeselector=dict(
                             buttons=list([
                                 dict(count=7, label="1w", step="day", stepmode="backward"),
                                 dict(count=1, label="1m", step="month", stepmode="backward"),
                                 dict(count=2, label="2m", step="month", stepmode="backward"),
                                 dict(count=3, label="3m", step="month", stepmode="todate"),
                                 dict(step="all")
                             ]))
                         )
    return fig_dyn


def create_fun(campaign):
    df_funnel = pd.read_sql(f'''''', con=conn)

    try:
        test_funnel = df_funnel[df_funnel['IS_CONTROL'] == 0][
            ['OVERALL', 'READ', 'TARGET_ACTION']].iloc[0]

        fig_funnel = go.Figure(go.Funnel(
            name='Тест',
            y=test_funnel.index,
            x=test_funnel.values,
            textposition="auto",
            textinfo="value+percent initial",
            marker_color='#0341fc'
        )
        )
        # fig_funnel.update_layout(title='Воронка')
        fig_funnel.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    except Exception as e:
        print(e)
        fig_funnel = go.Figure()

    return fig_funnel

def create_combined(date_aligned,radio):
    fig_dyn_all = go.Figure()

    for camp_id in dyn_all['CAMP_ID'].unique():
        dyn = dyn_all[dyn_all['CAMP_ID']==camp_id]
        prd_name = dyn['PRD_NAME'].iloc[0]
        cg_balance = cg_balance_all[cg_balance_all['CAMP_ID']==camp_id]
        cg_imbalance_rate = 1
        if len(cg_balance)>1:
                cg_imbalance_rate = cg_balance[cg_balance['IS_CONTROL']==0]['CNT'].iloc[0]/\
                                    cg_balance[cg_balance['IS_CONTROL']==1]['CNT'].iloc[0]

        test_dyn_y = dyn[dyn['IS_CONTROL']==0]['CNT']
        test_dyn_x = dyn[dyn['IS_CONTROL']==0]['TARGET_ACTION_DT']
        control_dyn_y = dyn[dyn['IS_CONTROL']==1]['CNT']*cg_imbalance_rate
        control_dyn_x = dyn[dyn['IS_CONTROL']==1]['TARGET_ACTION_DT']

        conv_test_dyn = dyn[dyn['IS_CONTROL']==0]['CNT'].cumsum()/cg_balance[cg_balance['IS_CONTROL']==0]['CNT'].iloc[0]
        conv_incr_test_dyn = conv_test_dyn.diff()
            
        if len(cg_balance)>1:
            conv_control_dyn = dyn[dyn['IS_CONTROL']==1]['CNT'].cumsum()/cg_balance[cg_balance['IS_CONTROL']==1]['CNT'].iloc[0]
            conv_incr_control_dyn = conv_control_dyn.diff()

        if radio == 'TAR_ACT':
            yt= test_dyn_y
            if len(cg_balance)>1:
                yc = control_dyn_y
            title ='Уникальных клиентов'
        elif radio == 'CONV_INCR':
            yt=conv_incr_test_dyn
            if len(cg_balance)>1:
                yc=conv_incr_control_dyn
            title = 'Прирост конверсии'

        if date_aligned:
            x_test = list(range(1,30))
            x_control = list(range(1,30))
        else:
            x_test = test_dyn_x
            x_control = control_dyn_x

        fig_dyn_all.add_trace(go.Scatter(x=x_test,
                                    y=yt,
                                    name=f'{prd_name}_{camp_id} Тест',
                                    line_shape='spline',
                                    mode='lines'
                                    ))
        if len(cg_balance)>1:
                fig_dyn_all.add_trace(go.Scatter(x=x_control,
                                        y=yc,
                                        name=f'{prd_name}_{camp_id} Контроль',
                                        line_shape='spline',
                                        mode='lines',
                                        line=dict(dash='dash')
                                        
                ))
    fig_dyn_all.update_yaxes(title=title)

    fig_dyn_all.update_layout(margin=dict(l=20, r=20, t=20, b=30), legend=dict(yanchor='top', xanchor='right'))

    fig_dyn_all.update_xaxes(title='Дата', rangeslider_visible=True,
                                rangeselector=dict(
                                    buttons=list([
                                        dict(count=7, label="1w", step="day", stepmode="backward"),
                                        dict(count=1, label="1m", step="month", stepmode="backward"),
                                        dict(count=2, label="2m", step="month", stepmode="backward"),
                                        dict(count=3, label="3m", step="month", stepmode="todate"),
                                        dict(step="all")
                                    ]))
                                )
            
    return fig_dyn_all


@app.callback(
    [Output('dyn', 'figure'),
     Output('fun', 'figure'),
     Output('info', 'children'),
     Output('metrics', 'children')],
    [Input('campaign', 'value'),
    Input('dyn_radio','value')])
def update_graph(campaign,value):
    a = time.time()
    metrics = create_metrics(campaign)
    b = time.time()
    cmp_info_html = create_info(campaign)
    c = time.time()
    fig_dyn = create_dyn(campaign,value)
    d=time.time()
    fig_funnel = create_fun(campaign)
    e = time.time()
    print(f'metrics time: {(b-a)}')
    print(f'info time: {(c-b)}')
    print(f'dynamic time: {(d-c)}')
    print(f'funnel time: {(e-d)}')
    return [fig_dyn, fig_funnel, cmp_info_html, metrics]

@app.callback(
    Output('fig_combined', 'figure'),
    [Input('date_align_switch', 'on'),
     Input('all_radio','value')])
def update_combined(on, value):
    fig_combined = create_combined(on, value)
    return fig_combined

if __name__ == '__main__':
    app.run_server(debug=True, port=8055, host='0.0.0.0')
