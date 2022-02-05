"""
Scripts for more interactive visualization of SEDs, etc.
"""
import numpy as np

from . import utils

__all__ = ['EazyExplorer']

class EazyExplorer(object):
    def __init__(self, photoz, zout, selection=None):
        """
        Generating a tool for interactive visualization of `eazy` outputs with 
        the `dash` + `plotly` libraries.
        """
        import pandas as pd
        try:
            import dash
            from dash import  dcc
            from dash import html 
            import plotly.express as px
            
        except ImportError:
            print('Failed to import dash & plotly, so the interactive tool'
                  'won\t work.\n'
                  'Install with `pip install dash==2.0` and also '
                  '`pip install jupyter_dash` for running a server '
                  'through jupyter')
                  
        uv = -2.5*np.log10(zout['restU']/zout['restV'])
        vj = -2.5*np.log10(zout['restV']/zout['restJ'])
        ssfr = zout['sfr']/zout['mass']

        df = pd.DataFrame()
        df['id'] = zout['id']
        df['nusefilt'] = zout['nusefilt']
        df['uv'] = uv
        df['vj'] = vj
        df['ssfr'] = np.log10(ssfr)
        df['mass'] = np.log10(zout['mass'])
        df['z_phot'] = zout['z_phot']
        df['z_spec'] = np.clip(photoz.ZSPEC*1, -0.1, 12)
        df['ra'] = photoz.RA
        df['dec'] = photoz.DEC
        df['chi2'] = zout['z_phot_chi2']/zout['nusefilt']
        if selection is not None:
            df = df[selection]
                
        _red_ix = np.argmax(photoz.pivot*(photoz.pivot < 3.e4))
        self.DEFAULT_FILTER = photoz.flux_columns[_red_ix]
        
        ZP = photoz.param['PRIOR_ABZP']*1.
        fmin = 10**(-0.4*(33-ZP))
        fmax = 10**(-0.4*(12-ZP))
        #print('flux limits', fmin, fmax)
        
        for f in photoz.flux_columns:
            key = f'mag_{f}'
            df[key] = ZP - 2.5*np.log10(np.clip(photoz.cat[f], fmin, fmax))
        
        df['mag'] = df[f'mag_{self.DEFAULT_FILTER}']
        
        self.df = df

        self.zout = zout
        self.photoz = photoz
        
        self.ZMAX = photoz.zgrid.max()
        self.MAXNFILT = photoz.nusefilt.max()
        

    @property
    def ra_bounds(self):
        return (self.df['ra'].max(), self.df['ra'].min())


    @property 
    def dec_bounds(self):
        return (self.df['dec'].min(), self.df['dec'].max())


    def make_dash_app(self, template='plotly_white', server_mode='external', port=8050, app_type='jupyter', plot_height=680, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']):
        """
        Create a Plotly/Dash app for interactive exploration
        
        Parameters
        ----------
        template : str
            `plotly` style `template <https://plotly.com/python/templates/#specifying-themes-in-graph-object-figures>`_.
        
        server_mode, port : str, int
            The app server is generated with 
            `app.run_server(mode=server_mode, port=port)`.
        
        app_type : str
            If ``jupyter`` then `app = jupyter_dash.JupyterDash()`, else
            `app = dash.Dash()`
            
        plot_height : int
            Height in pixels of the scatter and SED+P(z) plot windows.
            
        """
        import dash
        from dash import  dcc
        from dash import html 
        import plotly.express as px
        from urllib.parse import urlparse, parse_qsl, urlencode

        if app_type == 'dash':
            app = dash.Dash(__name__, 
                            external_stylesheets=external_stylesheets)
        else:
            from jupyter_dash import JupyterDash
            app = JupyterDash(__name__, 
                              external_stylesheets=external_stylesheets)

        PLOT_TYPES = ['zphot-zspec', 'Mag-redshift', 'Mass-redshift', 'UVJ', 
                      'RA/Dec']
                      
        COLOR_TYPES = ['z_phot', 'z_spec', 'mass', 'sSFR', 'chi2']
        
        #_title = f"{self.photoz.param['MAIN_OUTPUT_FILE']}"
        #_subhead = f"Nobj={self.photoz.NOBJ}  Nfilt={self.photoz.NFILT}"
        _title = [html.Strong(self.photoz.param['MAIN_OUTPUT_FILE']), 
                  ' / N', html.Sub('obj'), f'={self.photoz.NOBJ}', 
                  ' / N', html.Sub('filt'), f'={self.photoz.NFILT}', 
                  ]
        
        slider_row_style={'width': '90%', 'float':'left', 
                          'margin-left':'10px'}          
        slider_container = {'width': '150px', 'margin-left':'-25px'}
        check_kwargs = dict(style={'text-align':'center', 
                                   'height':'14pt', 
                                   'margin-top':'-20px'})
                                        
        ####### App layout
        app.layout = html.Div([
            # Selectors
            html.Div([
                dcc.Location(id='url', refresh=False), 

                html.Div([
                    html.Div(_title, id='title-bar', 
                             style={'float':'left', 'margin-top':'4pt'}),
                    
                    html.Div([
                        html.Div([dcc.Dropdown(id='plot-type',
                                     options=[{'label': i, 'value': i}
                                              for i in PLOT_TYPES],
                                     value='zphot-zspec', 
                                     style={'width':'160px', 
                                            'margin-right':'5px',
                                            'margin-left':'5px'}),
                        ], style={'float':'left'}),
                        
                        html.Div([dcc.Dropdown(id='color-type',
                                     options=[{'label': i, 'value': i}
                                              for i in COLOR_TYPES],
                                     value='sSFR', 
                                     style={'width':'100px', 
                                            'margin-right':'5px'}),
                        ], style={'display':'inline-block', 
                                  'margin-left':'10px'}),
                    ], style={'float':'right'}),
                ], style=slider_row_style),
                
                html.Div([
                    html.Div([dcc.Dropdown(id='mag-filter',
                                 options=[{'label': i, 'value': i}
                                          for i in self.photoz.flux_columns],
                                 value=self.DEFAULT_FILTER, 
                                 style={'width':'105px', 
                                        'margin-right':'20px'},
                                 clearable=False),
                    ], style={'float':'left'}),

                    html.Div([
                        dcc.RangeSlider(id='mag-slider',
                                        min=12, max=32, step=0.2,
                                        value=[18, 27],
                                        updatemode='drag',
                                        tooltip={"placement":'left'}), 

                        dcc.Checklist(id='mag-checked', 
                                      options=[{'label':'AB mag', 
                                                'value':'checked'}], 
                                      value=['checked'], **check_kwargs),
                              
                    ], style=dict(display='inline-block',
                                  **slider_container)),
                    #
                    html.Div([
                        dcc.RangeSlider(id='chi2-slider',
                                        min=0, max=10, step=0.1,
                                        value=[0, 3],
                                        updatemode='drag',
                                        tooltip={"placement":'left'}),

                        dcc.Checklist(id='chi2-checked', 
                                      options=[{'label':'chi2',
                                                'value':'checked'}], 
                                      value=[], **check_kwargs),

                    ], style=dict(display='inline-block', 
                                  **slider_container)),
                    
                    html.Div([
                        dcc.RangeSlider(id='nfilt-slider',
                                        min=1, max=self.MAXNFILT, step=1,
                                        value=[3, self.MAXNFILT],
                                        updatemode='drag',
                                        tooltip={"placement":'left'}),

                        dcc.Checklist(id='nfilt-checked', 
                                      options=[{'label':'nfilt', 
                                                'value':'checked'}], 
                                      value=['checked'], **check_kwargs),

                    ], style=dict(display='inline-block', 
                                  **slider_container)),

                ], style=slider_row_style),
                
                html.Div([
                    html.Div([
                        dcc.RangeSlider(id='zphot-slider',
                                        min=-0.5, max=12, step=0.1,
                                        value=[0, 6.5],
                                        updatemode='drag',
                                        tooltip={"placement":'left'}),
                          
                         dcc.Checklist(id='zphot-checked', 
                                       options=[{'label':'z_phot', 
                                                 'value':'checked'}], 
                                       value=['checked'], **check_kwargs),
                
                    ], style=dict(float='left', **slider_container)), 

                    html.Div([
                        dcc.RangeSlider(id='zspec-slider',
                                        min=-0.5, max=12, step=0.1,
                                        value=[-0.5, 6.5],
                                        updatemode='drag',
                                        tooltip={"placement":'left'}),
                          
                        dcc.Checklist(id='zspec-checked', 
                                      options=[{'label':'z_spec', 
                                                'value':'checked'}], 
                                      value=['checked'], **check_kwargs),

                    ], style=dict(display='inline-block',
                                  **slider_container)),

                    html.Div([
                        dcc.RangeSlider(id='mass-slider',
                                        min=7, max=13, step=0.1,
                                        value=[8, 11.8],
                                        updatemode='drag',
                                        tooltip={"placement":'left'}),

                        dcc.Checklist(id='mass-checked', 
                                      options=[{'label':'mass', 
                                                'value':'checked'}], 
                                      value=['checked'], **check_kwargs),

                    ], style=dict(display='inline-block', 
                                  **slider_container)),
                    
                ], style=slider_row_style), 

            ], style={'float':'left','width': '55%'}),

            # Object-level controls
            html.Div([
                html.Div([
                    html.Div('ID / RA,Dec.', style={'float':'left', 
                                                       'width':'100px', 
                                                       'margin-top':'5pt'}), 
                    
                    dcc.Input(id='id-input', type='text',
                          style={'width':'180px', 'padding':'2px', 
                                 'display':'inline'}), 
                    html.Div(children='', id='match-sep', 
                           style={'margin':'5pt', 'display':'inline', 
                                  'width':'50px'}),
                    
                    dcc.RadioItems(id='sed-unit-selector',
                                             options=[{'label': i, 'value': i}
                                             for i in ['Fλ', 'Fν', 'νFν']],
                                             value='Fλ',
                                             labelStyle={'display':'inline', 
                                                         'padding':'5px', 
                                                         },
                                             style={'display':'inline',
                                                    'width':'150px'})
                                                              
                ],  style={'width':'260pix', 'float':'left', 
                           'margin-right':'20px'}),
                
            ]), 
            html.Div([           
                html.Div([
                ],  style={'width':'120px', 'float':'left'}), 

                html.Div(id='object-info', children='ID: ', 
                        style={'display':'inline-block','margin-top':'10px'})

            ], style={'float':'right', 'width': '41%'}),

            # Plots
            html.Div([# Scatter plot
                dcc.Graph(id='sample-selection-scatter', 
                          hoverData={'points': [{'customdata':
                                             (self.df['id'][0], 1.0, -9.0)}]}, 
                          style={'width':'95%'})
            ], style={'float':'left', 'height':'70%', 'width':'49%'}), 

            html.Div([# SED
                dcc.Graph(id='object-sed-figure',
                          style={'width':'95%'})
            ], style={'float':'right', 'width':'49%', 'height':'70%'}),

        ])


        ##### Callback functions
        @app.callback(
             dash.dependencies.Output('url', 'search'),
            [dash.dependencies.Input('plot-type', 'value'),
             dash.dependencies.Input('color-type', 'value'),
             dash.dependencies.Input('mag-filter', 'value'),
             dash.dependencies.Input('mag-slider', 'value'),
             dash.dependencies.Input('mass-slider', 'value'),
             dash.dependencies.Input('chi2-slider', 'value'),
             dash.dependencies.Input('nfilt-slider', 'value'),
             dash.dependencies.Input('zphot-slider', 'value'),
             dash.dependencies.Input('zspec-slider', 'value'),
             dash.dependencies.Input('id-input', 'value')])
        def update_url_state(plot_type, color_type, mag_filter, mag_range, mass_range, chi2_range, nfilt_range, zphot_range, zspec_range, id_input):
            search = f'?plot_type={plot_type}&color_type={color_type}'
            search += f'&mag_filter={mag_filter}'
            search += f'&mag={mag_range[0]},{mag_range[1]}'
            search += f'&mass={mass_range[0]},{mass_range[1]}'
            search += f'&chi2={chi2_range[0]},{chi2_range[1]}'
            search += f'&nfilt={nfilt_range[0]},{nfilt_range[1]}'
            search += f'&zphot={zphot_range[0]},{zphot_range[1]}'
            search += f'&zspec={zspec_range[0]},{zspec_range[1]}'
            if id_input is not None:
                search += f"&id={id_input.replace(' ', '%20')}"
                
            return search


        @app.callback([dash.dependencies.Output('plot-type', 'value'),
                       dash.dependencies.Output('color-type', 'value'),
                       dash.dependencies.Output('mag-filter', 'value'),
                       dash.dependencies.Output('mag-slider', 'value'),
                       dash.dependencies.Output('mass-slider', 'value'),
                       dash.dependencies.Output('chi2-slider', 'value'),
                       dash.dependencies.Output('nfilt-slider', 'value'),
                       dash.dependencies.Output('zphot-slider', 
                                                'value'),
                       dash.dependencies.Output('zspec-slider', 
                                                'value'),
                       dash.dependencies.Output('id-input', 'value'),
                      ],[
                       dash.dependencies.Input('url', 'href')
                      ])
        def set_state_from_url(href):
            plot_type = 'zphot-zspec'
            color_type = 'sSFR'
            mag_filter = self.DEFAULT_FILTER
            mag_range = [18, 27]
            mass_range = [8, 11.6]
            chi2_range = [0, 4]
            nfilt_range = [1, self.MAXNFILT]
            zphot_range = [0, 6.5]
            zspec_range = [-0.5, 6.5]
            id_input = None

            if '?' not in href:
                return (plot_type, color_type, mag_filter, mag_range,
                        mass_range, chi2_range, nfilt_range,
                        zphot_range, zspec_range,
                        id_input)

            search = href.split('?')[1]
            params = search.split('&')

            for p in params:
                if 'plot_type' in p:
                    val = p.split('=')[1]
                    if val in PLOT_TYPES:
                        plot_type = val

                elif 'color_type' in p:
                    val = p.split('=')[1]
                    if val in COLOR_TYPES:
                        color_type = val
                        
                elif 'mag_filter' in p:
                    val = p.split('=')[1]
                    if val in self.photoz.flux_columns:
                        mag_filter = val
                    
                elif 'mag=' in p:
                    try:
                        vals = [float(v) for v in p.split('=')[1].split(',')]
                        if len(vals) == 2:
                            mag_range = vals
                    except ValueError:
                        pass
                        
                elif 'mass' in p:
                    try:
                        vals = [float(v) for v in p.split('=')[1].split(',')]
                        if len(vals) == 2:
                            mass_range = vals
                    except ValueError:
                        pass
                
                elif 'nfilt=' in p:
                    try:
                        vals = [int(v) for v in p.split('=')[1].split(',')]
                        if len(vals) == 2:
                            nfilt_range = vals
                    except ValueError:
                        pass
                
                elif 'zspec' in p:
                    try:
                        vals = [float(v) for v in p.split('=')[1].split(',')]
                        if len(vals) == 2:
                            zspec_range = vals
                    except ValueError:
                        pass
                        
                elif 'zphot' in p:
                    try:
                        vals = [float(v) for v in p.split('=')[1].split(',')]
                        if len(vals) == 2:
                            zphot_range = vals
                    except ValueError:
                        pass
                        
                elif 'id' in p:
                    try:
                        id_input = p.split('=')[1].replace('%20', ' ')
                    except ValueError:
                        id_input = None
                    
                    if not id_input:
                        id_input = None
                        
            return (plot_type, color_type, mag_filter, mag_range,
                    mass_range, chi2_range, nfilt_range,
                    zphot_range, zspec_range,
                    id_input)


        @app.callback(
            dash.dependencies.Output('sample-selection-scatter', 'figure'),
            [dash.dependencies.Input('plot-type', 'value'),
             dash.dependencies.Input('color-type', 'value'), 
             dash.dependencies.Input('mag-filter', 'value'),
             dash.dependencies.Input('mag-slider', 'value'),
             dash.dependencies.Input('mag-checked', 'value'),
             dash.dependencies.Input('mass-slider', 'value'),
             dash.dependencies.Input('mass-checked', 'value'),
             dash.dependencies.Input('chi2-slider', 'value'),
             dash.dependencies.Input('chi2-checked', 'value'),
             dash.dependencies.Input('nfilt-slider', 'value'),
             dash.dependencies.Input('nfilt-checked', 'value'),
             dash.dependencies.Input('zphot-slider', 'value'),
             dash.dependencies.Input('zphot-checked', 'value'),
             dash.dependencies.Input('zspec-slider', 'value'),
             dash.dependencies.Input('zspec-checked', 'value'),
             dash.dependencies.Input('id-input', 'value')])
        def update_selection(plot_type, color_type, mag_filter, mag_range, mag_checked, mass_range, mass_checked, chi2_range, chi2_checked, nfilt_range, nfilt_checked, zphot_range, zphot_checked, zspec_range, zspec_checked, id_input):
            """
            Apply slider selections
            """
            sel = np.isfinite(self.df['z_phot'])
            if 'checked' in zphot_checked:
                sel &= (self.df['z_phot'] > zphot_range[0]) 
                sel &= (self.df['z_phot'] < zphot_range[1])
            
            if 'checked' in zspec_checked:
                sel &= (self.df['z_spec'] > zspec_range[0]) 
                sel &= (self.df['z_spec'] < zspec_range[1])
            
            if 'checked' in mass_checked:
                sel &= (self.df['mass'] > mass_range[0])
                sel &= (self.df['mass'] < mass_range[1])

            if 'checked' in chi2_checked:
                sel &= (self.df['chi2'] >= chi2_range[0])
                sel &= (self.df['chi2'] <= chi2_range[1])

            if 'checked' in nfilt_checked:
                sel &= (self.df['nusefilt'] >= nfilt_range[0])
                sel &= (self.df['nusefilt'] <= nfilt_range[1])
            
            #print('redshift: ', sel.sum())
            
            if mag_filter is None:
                mag_filter = self.DEFAULT_FILTER

            #self.self.df['mag'] = self.ABZP 
            #self.self.df['mag'] -= 2.5*np.log10(self.photoz.cat[mag_filter])
            mag_col = 'mag_'+mag_filter            
            if 'checked' in mag_checked:
                sel &= (self.df[mag_col] > mag_range[0]) 
                sel &= (self.df[mag_col] < mag_range[1])
                
            self.df['mag'] = self.df[mag_col]
            
            #print('mag: ', sel.sum())
            
            if plot_type == 'zphot-zspec':
                sel &= self.df['z_spec'] > 0
            
            #print('zspec: ', sel.sum())
            
            if id_input is not None:
                id_i, dr_i = parse_id_input(id_input)
                if id_i is not None:
                    self.df['is_selected'] = self.df['id'] == id_i
                    sel |= self.df['is_selected']
                else:
                    self.df['is_selected'] = False
            else:
                self.df['is_selected'] = False

            dff = self.df[sel]
            
            # Color-coding by color-type pulldown
            if color_type == 'z_phot':
                color_kwargs = dict(color=np.clip(dff['z_phot'], 
                                                  *zphot_range),
                                    color_continuous_scale='portland')
            elif color_type == 'z_spec':
                color_kwargs = dict(color=np.clip(dff['z_spec'], 
                                                  *zspec_range), 
                                    color_continuous_scale='portland')
            elif color_type == 'mass':
                color_kwargs = dict(color=np.clip(dff['mass'], *mass_range), 
                                    color_continuous_scale='magma_r')
            elif color_type == 'chi2':
                color_kwargs = dict(color=np.clip(dff['chi2'], *chi2_range), 
                                    color_continuous_scale='balance')            
            else:
                color_kwargs = dict(color=np.clip(dff['ssfr'], -12., -8.), 
                                    color_continuous_scale='portland_r')
            
            # Scatter plot            
            if plot_type == 'Mass-redshift':
                args = ('z_phot','mass',
                        'z<sub>phot</sub>', 'log Stellar mass', 
                        (-0.1, self.ZMAX), (7.5, 12.5),
                        {}, color_kwargs)

            elif plot_type == 'Mag-redshift':
                args = ('z_phot','mag',
                        'z<sub>phot</sub>', f'AB mag ({mag_filter})', 
                        (-0.1, self.ZMAX), (18, 28),
                        {}, color_kwargs)

            elif plot_type == 'RA/Dec':
                args = ('ra','dec',
                        'R.A.', 'Dec.', 
                        self.ra_bounds, self.dec_bounds, 
                        {}, color_kwargs)

            elif plot_type == 'zphot-zspec':
                args = ('z_spec','z_phot',
                        'z<sub>spec</sub>', 'z<sub>phot</sub>', 
                        (0, 4.5), (0, 4.5), 
                        {}, color_kwargs)
            else:
                args = ('vj','uv',
                        '(V-J)', '(U-V)', 
                        (-0.1, 2.5), (-0.1, 2.5), 
                        {}, color_kwargs)

            fig = update_sample_scatter(dff, *args)

            if ('Mass' in plot_type) & ('checked' in mass_checked):
                fig.update_yaxes(range=mass_range)

            if ('Mag' in plot_type) & ('checked' in mag_checked):
                fig.update_yaxes(range=mag_range)
            
            if ('redshift' in plot_type) & ('checked' in zphot_checked):
                fig.update_xaxes(range=zphot_range)
            
            if ('zspec' in plot_type) & ('checked' in zspec_checked):
                    fig.update_yaxes(range=zspec_range)
            
            return fig


        def update_sample_scatter(dff, xcol, ycol, x_label, y_label, x_range, y_range, extra,  color_kwargs):
            """
            Make scatter plot
            """
            import plotly.graph_objects as go
            
            fig = px.scatter(data_frame=dff, x=xcol, y=ycol, 
                             custom_data=['id','z_phot','mass','ssfr','mag'], 
                             **color_kwargs)
                        
            htempl = '(%{x:.2f}, %{y:.2f}) <br>'
            htempl += 'id: %{customdata[0]:0d}  z_phot: %{customdata[1]:.2f}'
            htempl += '<br> mag: %{customdata[4]:.1f}  '
            htempl += 'mass: %{customdata[2]:.2f}  ssfr: %{customdata[3]:.2f}'

            fig.update_traces(hovertemplate=htempl, opacity=0.7)

            if dff['is_selected'].sum() > 0:
                dffs = dff[dff['is_selected']]
                _sel = go.Scatter(x=dffs[xcol], y=dffs[ycol],
                                  mode="markers+text",
                                  text=[f'{id}' for id in dffs['id']],
                                  textposition="bottom center",
                                  marker=dict(color='rgba(250,0,0,0.5)', 
                                              size=20, 
                                              symbol='circle-open'))
                                  
                fig.add_trace(_sel)
            
            fig.update_xaxes(range=x_range, title_text=x_label)
            fig.update_yaxes(range=y_range, title_text=y_label)

            fig.update_layout(template=template, 
                              autosize=True, showlegend=False, 
                              margin=dict(l=0,r=0,b=0,t=20,pad=0,
                                          autoexpand=True))
            
            if plot_height is not None:
                fig.update_layout(height=plot_height)
                
            fig.update_traces(marker_showscale=False, 
                              selector=dict(type='scatter'))
            fig.update_coloraxes(showscale=False)
            
            fig.add_annotation(text=f'N = {len(dff)} / {len(self.df)}',
                          xref="x domain", yref="y domain",
                          x=0.98, y=0.05, showarrow=False)
            
            return fig

        # @app.callback([dash.dependencies.Output('id-input', 'value')], 
        #               [dash.dependencies.Input('radec-input', 'value')])
        # def get_id_from_radec(radec_text):
        #     """
        #     Parse ra/dec text for nearest object
        #     """
        #     ra, dec = np.cast[float](radec_text.replace(',',' ').split())
        #     
        #     cosd = np.cos(self.df['dec']/180*np.pi)
        #     dx = (self.df['ra'] - ra)*cosd
        #     dy = (self.df['dec'] - dec)
        #     dr = np.sqrt(dx**2+dy**2)*3600.
        #     
        #     return df['id'][np.argmin(dr)]
        def parse_id_input(id_input):
            """
            Parse input as id or (ra dec)
            """
            if id_input in ['None', None, '']:
                return None, None
            
            inp_split = id_input.replace(',',' ').split()
            
            if len(inp_split) == 1:
                return int(inp_split[0]), None
                
            ra, dec = np.cast[float](inp_split)
            
            cosd = np.cos(self.df['dec']/180*np.pi)
            dx = (self.df['ra'] - ra)*cosd
            dy = (self.df['dec'] - dec)
            dr = np.sqrt(dx**2+dy**2)*3600.
            imin = np.nanargmin(dr)
            
            return self.df['id'][imin], dr[imin]
            

        @app.callback([dash.dependencies.Output('object-sed-figure', 
                                                'figure'),
                       dash.dependencies.Output('object-info', 'children'), 
                       dash.dependencies.Output('match-sep', 'children')], 
                      [dash.dependencies.Input('sample-selection-scatter', 
                                               'hoverData'), 
                       dash.dependencies.Input('sed-unit-selector', 'value'),
                       dash.dependencies.Input('id-input', 'value')])
        def update_object_sed(hoverData, sed_unit, id_input):
            """
            SED + p(z) plot
            """
            id_i, dr_i = parse_id_input(id_input)
            if id_i is None:
                id_i = hoverData['points'][0]['customdata'][0]
            else:
                if id_i not in self.zout['id']:
                    id_i = hoverData['points'][0]['customdata'][0]
            
            if dr_i is None:
                match_sep = ''
            else:
                match_sep = f'{dr_i:.1f}"'
                        
            show_fnu = {'Fλ':0, 'Fν':1, 'νFν':2}
            
            layout_kwargs = dict(template=template, 
                                 autosize=True, 
                                 showlegend=False, 
                                 margin=dict(l=0,r=0,b=0,t=20,pad=0,
                                               autoexpand=True))
                              
            fig = self.photoz.show_fit_plotly(id_i,
                                              show_fnu=show_fnu[sed_unit], 
                                              vertical=True, 
                                              panel_ratio=[0.6, 0.4],
                                              show=False,
                                              layout_kwargs=layout_kwargs)
            
            if plot_height is not None:
                fig.update_layout(height=plot_height)
            
            
            ix = self.df['id'] == id_i
            if ix.sum() == 0:
                object_info = 'ID: N/A'
            else:
                ix = np.where(ix)[0][0]
                ra, dec = self.df['ra'][ix], self.df['dec'][ix]
                #rd_links = query_html(ra, dec, queries=['CDS', 'ESO', 'MAST', 
                #                                        'LEG'])
                                                        
                object_info = [f'ID: {id_i}  |  α, δ) = {ra:.6f}, {dec:.6f} ',
                               html.A(' | ESO', 
                                      href=utils.eso_query(ra, dec, 
                                                           radius=1.0,
                                                           unit='s')),
                               html.A(' | CDS', 
                                      href=utils.cds_query(ra, dec, 
                                                           radius=1.0,
                                                           unit='s')),
                               html.A(' | Leg', 
                                      href=utils.show_legacysurvey(ra, dec, 
                                                           layer='dr9')),
                               html.Br(), 
                               f"z_phot: {self.df['z_phot'][ix]:.3f}  ", 
                               f" | z_spec: {self.df['z_spec'][ix]:.3f}", 
                               html.Br(),  
                               f"mag: {self.df['mag'][ix]:.2f}  ", 
                               f" | mass: {self.df['mass'][ix]:.2f} ",
                               f" | sSFR: {self.df['ssfr'][ix]:.2f}", 
                               html.Br()]

            return fig, object_info, match_sep

        app.run_server(mode=server_mode, port=port)
        return app    

