

"""# Dataset 1 (PH International Flights)"""

import pandas as pd

data = './datasets/PH_Airports_Arrivals_and_Departures.csv'
data = pd.read_csv(data)
#display(data.head())
#display(data.info())

data['firstSeen'] = pd.to_datetime(data['firstSeen'])
data['lastSeen'] = pd.to_datetime(data['lastSeen'])

# Define the start and end date for filtering
start_date = '2022-11-01'
end_date = '2023-10-31'

# Filter the dataset based on the 'firstSeen' or 'lastSeen' date range
filtered_data = data[(data['firstSeen'] >= start_date) & (data['firstSeen'] <= end_date) & (data['lastSeen'] >= start_date) & (data['lastSeen'] <= end_date)]

filtered_data.head()
filtered_data.info()

#display(filtered_data['estDepartureAirport'].isnull().sum(), filtered_data['estDepartureAirportHorizDistance'].isnull().sum(), filtered_data['estDepartureAirportVertDistance'].isnull().sum())
#display(filtered_data['estArrivalAirport'].isnull().sum(), filtered_data['estArrivalAirportHorizDistance'].isnull().sum(), filtered_data['estArrivalAirportVertDistance'].isnull().sum())

filtered_data_cleaned = filtered_data.dropna(subset=[
    'estDepartureAirport',
    'estDepartureAirportHorizDistance',
    'estDepartureAirportVertDistance',
    'estArrivalAirport',
    'estArrivalAirportHorizDistance',
    'estArrivalAirportVertDistance'
])

# Display the count of null values after removing rows
"""
display(
    filtered_data_cleaned['estDepartureAirport'].isnull().sum(),
    filtered_data_cleaned['estDepartureAirportHorizDistance'].isnull().sum(),
    filtered_data_cleaned['estDepartureAirportVertDistance'].isnull().sum()
)
display(
    filtered_data_cleaned['estArrivalAirport'].isnull().sum(),
    filtered_data_cleaned['estArrivalAirportHorizDistance'].isnull().sum(),
    filtered_data_cleaned['estArrivalAirportVertDistance'].isnull().sum()
)
"""
filtered_data_cleaned.info()

is_arrival_zero_count = (filtered_data_cleaned['IsArrival'] == 0).sum()

# Display the count
print(f"Number of entries where IsArrival = 0: {is_arrival_zero_count}")

data = filtered_data_cleaned[filtered_data_cleaned['IsArrival'] != 0]
data.info()

final = data.drop(columns=['icao24', 'callsign', 'beginWeek', 'endWeek', 'departureAirportCandidatesCount', 'arrivalAirportCandidatesCount', 'estDepartureAirportHorizDistance', 'estDepartureAirportVertDistance',
                           'estArrivalAirportHorizDistance','estArrivalAirportVertDistance', 'IsArrival']).reset_index(drop=True)
final.info()
final.head()
#final

final.duplicated().sum()

departure_values = final['estDepartureAirport'][final['estDepartureAirport'].str.startswith('RP')]
arrival_values = final['estArrivalAirport'][final['estArrivalAirport'].str.startswith('RP')]

# Combine both values
rp_values = pd.concat([departure_values, arrival_values])
print(rp_values.unique())

final.to_csv('PH International Flights.csv', index=False)

"""# Dataset 2 (PH Airports)"""

#import pandas as pd

df = './datasets/ph-airports (1).csv'
df = pd.read_csv(df)
#display(df.head())
df.info()

finaldf = df.drop(columns=['id', 'continent', 'country_name', 'iso_country', 'iso_region', 'wikipedia_link', 'keywords', 'score',
                           'last_updated', 'local_region', 'local_code', 'iata_code', 'home_link']).reset_index(drop=True)
finaldf = df.drop(df[(df['type'] == 'heliport') | (df['type'] == 'closed') | (df['type'] == 'balloonport') |
                  (df['type'] == 'seaplane_base')].index)
finaldf = finaldf.drop(index=finaldf.index[0]).reset_index(drop=True)
finaldf.info()
#display(finaldf.head())

rp_values = finaldf['ident'][finaldf['ident'].str.startswith('RP')]

print(rp_values.unique())

finaldf['type'].unique()

finaldf.to_csv('PH Airports.csv', index=False)

"""# Dataset 3 (Passenger, Cargo, Aircraft Movement)"""

#import pandas as pd

# Importing and cleaning the data
ddf = './datasets/PassengerMovement_2023.csv'
ddf = pd.read_csv(ddf)
ddf = ddf.fillna(0)
#display(ddf)
ddf.info()

import copy


def uncomma(s):
    parts = s.split(',')
    new_s = ""
    for p in parts:
        new_s = new_s + p
    return new_s

k = len(ddf['NOVEMBER'])

nov = copy.deepcopy(ddf['NOVEMBER'])
dec = copy.deepcopy(ddf['DECEMBER'])

for i in range(k):
    s_num = str(ddf['NOVEMBER'][i])
    n_num = uncomma(s_num)
    nov[i] = n_num

    s_num = str(ddf['DECEMBER'][i])
    n_num = uncomma(s_num)
    dec[i] = n_num

ddf['NOVEMBER'] = nov
ddf['DECEMBER'] = dec

ddf['JULY'][182] = 0
ddf['AUGUST'][182] = 0
ddf['SEPTEMBER'][182] = 0

ddf = ddf.astype({'NOVEMBER' : float, 'DECEMBER' : float, 'JULY' : float, 'AUGUST' : float, 'SEPTEMBER' : float,})
ddf = ddf.drop(ddf.index[185])

#display(ddf)
ddf.info()

dfc = './datasets/CargoMovement_2023.csv'
dfc = pd.read_csv(dfc)
dfc = dfc.fillna(0)
#display(dfc)
dfc.info()

k = len(dfc['NOVEMBER'])

nov = copy.deepcopy(dfc['NOVEMBER'])
dec = copy.deepcopy(dfc['DECEMBER'])

for i in range(k):
    s_num = str(dfc['NOVEMBER'][i])
    n_num = uncomma(s_num)
    nov[i] = n_num

    s_num = str(dfc['DECEMBER'][i])
    n_num = uncomma(s_num)
    dec[i] = n_num

dfc['NOVEMBER'] = nov
dfc['DECEMBER'] = dec

dfc['JULY'][130] = 0
dfc['AUGUST'][130] = 0
dfc['SEPTEMBER'][130] = 0

dfc = dfc.astype({'NOVEMBER' : float, 'DECEMBER' : float, 'JULY' : float, 'AUGUST' : float, 'SEPTEMBER' : float,})
dfc = dfc.drop(dfc.index[133])

#display(dfc)
dfc.info()

dfa = './datasets/AircraftMovement_2023.csv'
dfa = pd.read_csv(dfa)
dfa = dfa.fillna(0)
#display(dfa)
dfa.info()

dfa['JULY'][182] = 0
dfa['AUGUST'][182] = 0
dfa['SEPTEMBER'][182] = 0

dfa = dfa.astype({'JULY' : float, 'AUGUST' : float, 'SEPTEMBER' : float,})
dfa = dfa.drop(dfa.index[185])

#display(dfa)
dfa.info()

unique_p = ddf['Airport'].unique()
print(len(unique_p))
#display(unique_p)

unique_c = dfc['Airport'].unique()
print(len(unique_c))
#display(unique_c)

unique_a = dfa['Airport'].unique()
print(len(unique_a))
#display(unique_a)

# Grouping the data
def ignore_dom_intl(s):
    s_norm = s.lower()
    if s_norm.find("int'l") != -1 or s_norm.find("intl") != -1 or s_norm.find("dom") != -1:
        s_split = s.split(' ')
        return s_split[0]
    return s

ddf['airport_simple'] = ddf['Airport'].apply(ignore_dom_intl)
#display(ddf)
unique_p2 = ddf['airport_simple'].unique()
print(len(unique_p2))
#display(unique_p2)

dfc['airport_simple'] = dfc['Airport'].apply(ignore_dom_intl)
#display(dfc)
unique_c2 = dfc['airport_simple'].unique()
print(len(unique_c2))
#display(unique_c2)

dfa['airport_simple'] = dfa['Airport'].apply(ignore_dom_intl)
#display(dfa)
unique_a2 = dfa['airport_simple'].unique()
print(len(unique_a2))
#display(unique_a2)

ddf_group = ddf.drop(['Airport','AIRLINE\nOPERATOR'], axis = 1)
ddf_group = ddf_group.groupby(['airport_simple']).sum()
#display(ddf_group)

dfc_group = dfc.drop(['Airport','AIRLINE OPERATOR'], axis = 1)
dfc_group = dfc_group.groupby(['airport_simple']).sum()
#display(dfc_group)

dfa_group = dfa.drop(['Airport','AIRLINE\nOPERATOR'], axis = 1)
dfa_group = dfa_group.groupby(['airport_simple']).sum()
#display(dfa_group)

# Linking with airports dataset
ddf_mapping = {'Baguio': 'RPUB', 'Laoag': 'RPLI', 'Vigan': 'RPUQ', 'Bagabag': 'RPUZ', 'Basco': 'RPUO', 'Cauayan': 'RPUY',
       'Itbayat': 'RPLT', 'Palanan': 'RPLN', 'Tuguegarao': 'RPUT', 'Sangley': 'RPLS', 'Manila':'RPLL', 'Busuanga': 'RPVV', 'Cuyo': 'RPLO',
       'Marinduque': 'RPUW', 'Pto.Princesa': 'RPVP', 'Romblon': 'RPVU', 'San Jose': 'RPUH', 'San Vicente': 'RPSV',
       'Bicol': 'RPLK', 'Masbate': 'RPVJ', 'Naga': 'RPUN', 'Virac': 'RPUV', 'Antique': 'RPVS', 'Bacolod': 'RPVB',
       'Caticlan': 'RPVE', 'Iloilo': 'RPVI', 'Kalibo': 'RPVK', 'Roxas': 'RPVR', 'Dumaguete': 'RPVD',
       'Bohol-Panglao': 'RPSP', 'Borongan': 'RPVW', 'Calbayog': 'RPVC', 'Catarman': 'RPVF', 'Guiuan': 'RPVG',
       'Hilongos': 'RPVH', 'Maasin': 'RPSM', 'Ormoc': 'RPVO', 'Tacloban': 'RPVA', 'Dipolog': 'RPMG', 'Jolo': 'RPMJ',
       'Pagadian': 'RPMP', 'Sanga-Sanga': 'RPMN', 'Zamboanga': 'RPMZ', 'Butuan': 'RPME', 'Laguindingan': 'RPMY',
       'Camiguin': 'RPMH', 'Ozamis': 'RPMO', 'Siargao': 'RPNS', 'Surigao': 'RPMS', 'Gen. San.Tambler': 'RPMR',
       'Davao': 'RPMD', 'Cotabato': 'RPMC', 'Jomalig':'RPLJ', 'Catbalogan':'RPVY'}

dfc_mapping = {'Baguio': 'RPUB', 'Laoag': 'RPLI', 'Vigan': 'RPUQ', 'Basco': 'RPUO', 'Cauayan': 'RPUY',
       'Itbayat': 'RPLT', 'Palanan': 'RPLN', 'Tuguegarao': 'RPUT', 'Manila':'RPLL', 'Busuanga': 'RPVV', 'Cuyo': 'RPLO',
       'Marinduque': 'RPUW', 'Pto.Princesa': 'RPVP', 'Romblon': 'RPVU', 'San Jose': 'RPUH', 'San Vicente': 'RPSV',
       'Bicol': 'RPLK', 'Masbate': 'RPVJ', 'Naga': 'RPUN', 'Virac': 'RPUV', 'Antique': 'RPVS', 'Bacolod': 'RPVB',
       'Caticlan': 'RPVE', 'Iloilo': 'RPVI', 'Kalibo': 'RPVK', 'Roxas': 'RPVR', 'Dumaguete': 'RPVD',
       'Bohol-Panglao': 'RPSP', 'Borongan': 'RPVW', 'Catarman': 'RPVF', 'Guiuan': 'RPVG',
       'Maasin': 'RPSM', 'Ormoc': 'RPVO', 'Tacloban': 'RPVA', 'Dipolog': 'RPMG',
       'Pagadian': 'RPMP', 'Sanga-Sanga': 'RPMN', 'Zamboanga': 'RPMZ', 'Butuan': 'RPME', 'Laguindingan': 'RPMY',
       'Camiguin': 'RPMH', 'Ozamis': 'RPMO', 'Siargao': 'RPNS', 'Gen. San.Tambler': 'RPMR',
       'Davao': 'RPMD', 'Cotabato': 'RPMC'}

dfa_mapping = {'Baguio': 'RPUB', 'Laoag': 'RPLI', 'Vigan': 'RPUQ', 'Bagabag': 'RPUZ', 'Basco': 'RPUO', 'Cauayan': 'RPUY',
       'Itbayat': 'RPLT', 'Palanan': 'RPLN', 'Tuguegarao': 'RPUT', 'Sangley': 'RPLS', 'Manila':'RPLL', 'Busuanga': 'RPVV', 'Cuyo': 'RPLO',
       'Marinduque': 'RPUW', 'Pto.Princesa': 'RPVP', 'Romblon': 'RPVU', 'San Jose': 'RPUH', 'San Vicente': 'RPSV',
       'Bicol': 'RPLK', 'Masbate': 'RPVJ', 'Naga': 'RPUN', 'Virac': 'RPUV', 'Antique': 'RPVS', 'Bacolod': 'RPVB',
       'Caticlan': 'RPVE', 'Iloilo': 'RPVI', 'Kalibo': 'RPVK', 'Roxas': 'RPVR', 'Dumaguete': 'RPVD',
       'Bohol-Panglao': 'RPSP', 'Borongan': 'RPVW', 'Calbayog': 'RPVC', 'Catarman': 'RPVF', 'Guiuan': 'RPVG',
       'Hilongos': 'RPVH', 'Maasin': 'RPSM', 'Ormoc': 'RPVO', 'Tacloban': 'RPVA', 'Dipolog': 'RPMG', 'Jolo': 'RPMJ',
       'Pagadian': 'RPMP', 'Sanga-Sanga': 'RPMN', 'Zamboanga': 'RPMZ', 'Butuan': 'RPME', 'Laguindingan': 'RPMY',
       'Camiguin': 'RPMH', 'Ozamis': 'RPMO', 'Siargao': 'RPNS', 'Surigao': 'RPMS', 'Gen. San.Tambler': 'RPMR',
       'Davao': 'RPMD', 'Cotabato': 'RPMC', 'Jomalig':'RPLJ', 'Catbalogan':'RPVY'}

port_to_move = {}

for k in ddf_mapping.keys():
    icao = ddf_mapping[k]
    port_to_move[icao] = k

full_move = ddf_mapping.keys()

print("passenger:")
for port in unique_p2:
    if port not in full_move:
        print(port)
print("cargo:")
for port in unique_c2:
    if port not in full_move:
        print(port)
print("aircraft:")
for port in unique_a2:
    if port not in full_move:
        print(port)
print("\n")
print(len(ddf_mapping))
#display(port_to_move)

df_hist2 = ddf_group
airport_code = 'RPUB'
icao_port = port_to_move[airport_code]
column1 = df_hist2['JANUARY'][icao_port]
#display(column1)

#import plotly.graph_objects as go

"""# Dash App"""

#! pip install dash

from dash import Dash, html, dash_table, dcc
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px
from dash.dependencies import Input, Output
import plotly.graph_objects as go
#import matplotlib.pyplot as plt
#from matplotlib.widgets import Cursor

airports = finaldf
routes = pd.read_csv('./datasets/PH International Flights.csv')

airports['elevation_ft'] = pd.to_numeric(airports['elevation_ft']).fillna(0)

unique_routes = {}

for _, row in routes.iterrows():
    departure = row['estDepartureAirport']
    arrival = row['estArrivalAirport']
    if pd.notna(departure) and pd.notna(arrival):
        # Ensure no reverse duplicates
        unique_routes.setdefault(departure, {})
        unique_routes.setdefault(arrival, {})

        # Track forward and reverse connections
        unique_routes[departure][arrival] = unique_routes[departure].get(arrival, False) or True
        unique_routes[arrival][departure] = unique_routes[arrival].get(departure, False) or False

def get_route_status(departure, arrival):
    is_forward = unique_routes.get(departure, {}).get(arrival, False)
    is_reverse = unique_routes.get(arrival, {}).get(departure, False)

    if is_forward and is_reverse:
        return "Back-to-back"
    return "One-way"

size_mapping = {
    'large_airport': 30,
    'medium_airport': 20,
    'small_airport': 10,
}

airports.loc[:,'size'] = airports['type'].map(size_mapping).fillna(10)

def generate_hover_text(row):
    airport_code = row['ident']

    return (f"Airport Name: {row['name']}<br>"
            f"Ident: {row['ident']}<br>"
            f"Type: {row['type']}<br>"
            f"Elevation (ft): {row['elevation_ft']}<br>"
            f"Region: {row['region_name']}<br>"
            f"Municipality: {row['municipality']}<br>")

airports['hover_text'] = airports.apply(generate_hover_text, axis=1)

philippine_flight_counts = {airport: 0 for airport in unique_routes if airport.startswith('RP')}

for _, row in routes.iterrows():
    departure = row['estDepartureAirport']
    arrival = row['estArrivalAirport']
    if pd.notna(departure) and pd.notna(arrival):
        # Increment count for departure and arrival airports if they are Philippine airports
        if departure.startswith('RP'):
            philippine_flight_counts[departure] += 1
        if arrival.startswith('RP'):
            philippine_flight_counts[arrival] += 1

# Sort and select the top 5 Philippine airports
top_5_philippine_airports = sorted(philippine_flight_counts.items(), key=lambda x: x[1], reverse=True)[:5]

top_airports = [airport for airport, count in top_5_philippine_airports]
flight_counts = [count for airport, count in top_5_philippine_airports]

app = Dash(__name__)
map_fig = go.Figure()

map_fig.add_trace(go.Scattermapbox(
    lat=airports['latitude_deg'],
    lon=airports['longitude_deg'],
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=airports['size'],
        color='#133E87',
        opacity=0.7
    ),
    text=airports['hover_text'],
    hoverinfo='text'
))

map_fig.update_layout(
    mapbox=dict(
        accesstoken='pk.eyJ1IjoicGF0cmljaWFhbmgiLCJhIjoiY200NG8yOHY5MDExMjJqcHpzbXpkeWRsayJ9.X6ZGojlyHZQhtWLir8Idng',  # Replace with your Mapbox token
        center=dict(lat=12.8797, lon=121.7740),
        zoom=6,
        style='open-street-map'
    ),
    margin={"r": 0, "t": 0, "l": 0, "b": 0}
)

bar_fig = go.Figure(
    data=[
        go.Bar(
            x=top_airports,
            y=flight_counts,
            text=flight_counts,
            textposition='auto',
            marker=dict(color='#133E87', opacity=0.7),
        )
    ]
)

# Update layout
bar_fig.update_layout(
    title=" ",
    xaxis=dict(title="Airport"),
    yaxis=dict(title="Number of Flights"),
    margin=dict(l=40, r=40, t=40, b=40),
    paper_bgcolor='#F3F3E0'
)

# Define the months as an ordered list of month labels
month_labels = [
    "Nov\n2022", "Dec\n2022", "Jan\n2023", "Feb\n2023", "Mar\n2023", "Apr\n2023",
    "May\n2023", "Jun\n2023", "Jul\n2023", "Aug\n2023", "Sep\n2023", "Oct\n2023"
]

# Create a dictionary for the marks
marks = {i: month_labels[i] for i in range(12)}

app.layout = html.Div(
    [
        # Title Section
        html.Div(
            [
                html.H1(
                    "Visualization of the Performance of Philippine Airports",
                    style={
                        'textAlign': 'left',
                        'color': '#133E87',
                        'margin': '20px 20px',
                        'fontSize': '28px',
                    },
                ),
                html.P(
                    """
                    International tourism is a major part of the Philippine economy.
                    In 2023, tourism contributed 8.6% to the country’s GDP and employed 6.21 million Filipinos in its industry.
                     [1] However, this is still lower than the peak 12.7% contribution recorded before the COVID-19 pandemic [2].
                    Despite reaching 80% of the foreign tourist target in 2023 [3], it is acceptable to say that the industry has
                    seen better days. This visualization will focus on airports within the Philippines, however, some airports
                    are missing data which will result in blank graphs.
                    """,
                    style={
                        'textAlign': 'left',
                        'color': '#133E87',
                        'margin': '0 20px 20px',
                        'fontSize': '16px',
                        'lineHeight': '1.5',
                    },
                ),
            ],
            style={
                'background-color': '#F3F3E0',
                'border-bottom': '2px solid #133E87',
                'padding-bottom': '25px',
                'padding-top': '25px',
            },
        ),
        # Upper Section
        html.Div(
            [
                # Map and Routes Table Section
                html.Div(
                    children=[
                        dcc.Graph(
                            id="map",
                            figure=map_fig,
                            config={'scrollZoom': True},
                            style={
                                'height': '100%',
                                'border': '2px solid #133E87',
                                'background-color': '#F3F3E0',
                            },
                        ),
                        html.H2(
                            "Routes of Selected Airport",
                            style={
                                'textAlign': 'center',
                                'color': '#133E87',
                                'margin': '10px 0',
                            },
                        ),
                        dash_table.DataTable(
                            id='routes-table',
                            columns=[
                                {'name': 'Departure', 'id': 'departure'},
                                {'name': 'Arrival', 'id': 'arrival'},
                                {'name': 'Status', 'id': 'status'},
                            ],
                            data=[],
                            style_table={
                                'height': '300px',
                                'overflowY': 'auto',
                                # 'background-color': '#CBDCEB',
                            },
                            style_header={
                                'backgroundColor': '#133E87',  # Set background color of the header
                                'fontWeight': 'bold',  # Set font weight for the header
                                'textAlign': 'center'  # Align text to center
                            },
                            style_cell={
                                'textAlign': 'center',
                                'padding': '10px',
                                'fontFamily': 'Arial, sans-serif',
                                'color': '#F3F3E0',
                            },
                            style_data_conditional=[
                            {
                                'if': {'row_index': 'even'},  # Even rows
                                'backgroundColor': '#608BC1'
                            },
                            {
                                'if': {'row_index': 'odd'},  # Odd rows
                                'backgroundColor': '#133E87'
                            }
                         ]
                        ),
                    ],
                    style={
                        'width': '70%',
                        'height': '40%',
                        'display': 'inline-block',
                        'background-color': '#CBDCEB',
                    },
                ),
                # Top Airports Section
                html.Div(
                    [
                        html.Div(
                            children=[
                                html.H3(
                                    "Top 5 Philippine Airports with the Most Flights",
                                    style={
                                        'textAlign': 'center',
                                        'color': '#133E87',
                                        'padding': '5px',
                                    },
                                ),
                                dcc.Graph(
                                    id='bar-graph',
                                    figure=bar_fig,
                                    style={'height': '400px'},
                                ),
                            ],
                            style={
                                'height': '65%',
                                'background-color': '#F3F3E0',
                                'border': '2px solid #133E87',
                            },
                        ),
                    ],
                    style={
                        'width': '30%',
                        'display': 'inline-block',
                    },
                ),
            ],
            style={
                'display': 'flex',
                'flex-direction': 'row',
                'padding-left': '100px',
                'padding-right': '100px',
                'padding-top': '25px',
                'padding-bottom': '25px',
                'background-color': '#F3F3E0',
            },
        ),
        # Lower Section
        html.Div(
            [
                # Time Period Slider
                html.Div(
                    [
                        html.H3(
                            "Select Time Period",
                            style={
                                'color': '#133E87',
                                'margin-bottom': '10px',
                            },
                        ),
                        dcc.RangeSlider(
                            id='time-period-slider',
                            min=0,
                            max=11,
                            step=1,
                            value=[0, 11],
                            marks=marks,
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True,
                            },
                        ),
                    ],
                    style={
                        'width': '75%',
                        'height': '100px',
                        'padding': '20px',
                        'display': 'inline-block',
                        'background-color': '#CBDCEB',
                    },
                ),
                # Flight Type Dropdown
                html.Div(
                    [
                        html.H3(
                            "Select Flight Type",
                            style={
                                'color': '#133E87',
                                'margin-bottom': '10px',
                            },
                        ),
                        dcc.Dropdown(
                            id="dropdown",
                            options=[
                                {'label': 'Passenger', 'value': 'Passenger'},
                                {'label': 'Cargo', 'value': 'Cargo'},
                                {'label': 'Aircraft', 'value': 'Aircraft'},
                            ],
                            value='Passenger',
                            style={
                                'background-color': '#F3F3E0',
                                'color': '#133E87',
                            },
                        ),
                    ],
                    style={
                        'width': '17%',
                        'height': '100px',
                        'padding': '20px',
                        'display': 'inline-block',
                        'background-color': '#CBDCEB',
                    },
                ),
                # Time Histograms
                html.Div(
                    [
                        html.Div(
                            dcc.Graph(id='time-hist1'),
                            style={
                                'width': '50%',
                                'display': 'inline-block',
                                'background-color': '#CBDCEB',
                                'border': '2px solid #133E87',
                                'padding': '10px',
                            },
                        ),
                        html.Div(
                            dcc.Graph(id='time-hist2'),
                            style={
                                'width': '50%',
                                'display': 'inline-block',
                                'background-color': '#CBDCEB',
                                'border': '2px solid #133E87',
                                'padding': '10px',
                            },
                        ),
                    ],
                    style={
                        'display': 'flex',
                        'flex-direction': 'row',
                        'padding': '20px',
                        'background-color': '#F3F3E0',
                    },
                ),
            ],
            style={
                'background-color': '#F3F3E0',
                'border-top': '2px solid #133E87',
                'padding-top': '25px',
            },
        ),
    ],
    style={
        'font-family': 'Arial, sans-serif',
        'background-color': '#F3F3E0',
        'padding': '0 20px',
    },
)

@app.callback(
    Output('routes-table', 'data'),
     [Input('map', 'clickData'),
     Input('map', 'hoverData')]
)
def update_routes_table(clickData, hoverData):
    """
    if clickData is None and hoverData is None:
        return []  # Return empty if no clickData and hoverData
    """
    if clickData is None:
        return []  # Return empty if no clickData and hoverData

    if clickData:

    # Extract airport identifier from hover data
        airport_code = clickData['points'][0]['text'].split('<br>')[1].split(': ')[1]

    # Find routes for this airport
        connected_routes = unique_routes.get(airport_code, {})
        route_data = []

        for arrival, _ in connected_routes.items():
            if airport_code < arrival:  # Ensure unique pairs
                status = get_route_status(airport_code, arrival)
                route_data.append({'departure': airport_code, 'arrival': arrival, 'status': status})

    """
    elif hoverData:

    # Extract airport identifier from hover data
        airport_code = hoverData['points'][0]['text'].split('<br>')[1].split(': ')[1]

    # Find routes for this airport
        connected_routes = unique_routes.get(airport_code, {})
        route_data = []

        for arrival, _ in connected_routes.items():
            if airport_code < arrival:  # Ensure unique pairs
                status = get_route_status(airport_code, arrival)
                route_data.append({'departure': airport_code, 'arrival': arrival, 'status': status})
    """

    return route_data

@app.callback(
    Output('time-hist1', 'figure'),
    [Input('map', 'clickData'),
     Input('map', 'hoverData'),
     Input('time-period-slider','value')]
)
def update_histogram(clickData,hoverData,slider_value):
    """
    if clickData is None and hoverData is None:
        airport_code='RPLL'
    """
    if clickData is None:
        airport_code='RPLL'
    elif clickData:
        airport_code = clickData['points'][0]['text'].split('<br>')[1].split(': ')[1]

    """
    elif hoverData:
        airport_code = hoverData['points'][0]['text'].split('<br>')[1].split(': ')[1]
    """

    # Filter data for the selected airport
    filtered_df = routes[
        (routes['estDepartureAirport'] == airport_code) |
        (routes['estArrivalAirport'] == airport_code)
    ]
    filtered_df['firstSeen']= pd.to_datetime(filtered_df['firstSeen'])

    start_month, end_month = slider_value

    # Dataset-specific date bounds
    dataset_start = pd.Timestamp("2022-11-01")
    dataset_end = pd.Timestamp("2023-10-31")

    # Compute bounds from slider defined values
    start_date = dataset_start + pd.DateOffset(months=start_month)
    end_date = dataset_start + pd.DateOffset(months=end_month+1) - pd.Timedelta(days=1)

    """
    # Map slider values to corresponding months in the dataset
    start_date = pd.Timestamp(f"2022-{start_month + 1:02d}-01")
    end_date = pd.Timestamp(f"2022-{end_month + 1:02d}-01") + pd.DateOffset(months=1) - pd.Timedelta(days=1)

    # Adjust for wrap-around (November-December span to following year)
    if start_month > end_month:
        start_date = pd.Timestamp(f"2022-{start_month + 1:02d}-01")
        end_date = pd.Timestamp(f"2023-{end_month + 1:02d}-01") + pd.DateOffset(months=1) - pd.Timedelta(days=1)
    """

    # Clip to the dataset's actual date range
    start_date = max(start_date, dataset_start)
    end_date = min(end_date, dataset_end)

    # Filter the data based on the adjusted date range
    filtered_df = filtered_df[(filtered_df['firstSeen'] >= start_date) & (filtered_df['firstSeen'] <= end_date)]

    # Calculate the time difference for determining granularity
    time_diff = (end_date - start_date).days

    # Default: show monthly data if time range is greater than 3 months
    if time_diff > 90:
        filtered_df['month_year'] = filtered_df['firstSeen'].dt.to_period('M')
        monthly_flights = filtered_df.groupby('month_year').size()
        months = monthly_flights.index.to_timestamp()
        fig = go.Figure(data=[go.Bar(
            x=months,
            y=monthly_flights,
            text=[f"{period.strftime('%b %Y')}: {count}" for period, count in zip(months, monthly_flights)],
            textposition='outside',
            marker_color='#133E87'
        )])

    elif time_diff <= 90 and time_diff > 31:  # Weekly data
        filtered_df['week'] = filtered_df['firstSeen'].dt.to_period('W')
        weekly_flights = filtered_df.groupby('week').size()
        weeks = weekly_flights.index.to_timestamp()
        fig = go.Figure(data=[go.Bar(
            x=weeks,
            y=weekly_flights,
            text=[f"Week {period.strftime('%Y-%m-%d')}: {count}" for period, count in zip(weeks, weekly_flights)],
            textposition='outside',
            marker_color='#133E87'
        )])

    else:  # Daily data
        filtered_df['day'] = filtered_df['firstSeen'].dt.date
        daily_flights = filtered_df.groupby('day').size()
        days = pd.to_datetime(daily_flights.index)
        fig = go.Figure(data=[go.Bar(
            x=days,
            y=daily_flights,
            text=[f"{day.strftime('%Y-%m-%d')}: {count}" for day, count in zip(days, daily_flights)],
            textposition='outside',
            marker_color='#133E87'
        )])

    # Update layout for the figure
    fig.update_layout(
        title=f"Number of Flights Per Time Period for {airport_code}",
        xaxis_title="Date",
        yaxis_title="Number of Flights",
        xaxis=dict(tickformat="%b %Y" if 'month_year' in locals() else "%Y-%m"),
        yaxis=dict(showgrid=True),
        bargap=0.2,
        height=500,
        paper_bgcolor='#F3F3E0'
    )

    return fig

@app.callback(
    Output('time-hist2', 'figure'),
    [Input('map', 'clickData'),
     Input('map', 'hoverData'),
     Input('dropdown','value')]
)
def update_histogram2(clickData,hoverData,dropdown_value):
    """
    if clickData is None and hoverData is None:
        airport_code='RPLL'
    """
    if clickData is None:
        airport_code='RPLL'
    elif clickData:
        airport_code = clickData['points'][0]['text'].split('<br>')[1].split(': ')[1]

    """
    elif hoverData:
        airport_code = hoverData['points'][0]['text'].split('<br>')[1].split(': ')[1]
    """

    # Selecting dataset based on dropdown
    if dropdown_value == "Passenger":
        df_hist2 = ddf_group
    elif dropdown_value == "Cargo":
        df_hist2 = dfc_group
    else:
        df_hist2 = dfa_group

    # Selecting data based on hovered airport
    icao_port = ""
    dict_hist2 = {}

    if airport_code in port_to_move:
        icao_port = port_to_move[airport_code]

        if icao_port in df_hist2.index:

            for mon in df_hist2.columns:
                if mon == 'NOVEMBER' or mon == 'DECEMBER':
                    mon_y = mon + ' 2022'
                elif mon == 'TOTAL':
                    continue
                else:
                    mon_y = mon + ' 2023'

                dict_hist2[mon_y] = df_hist2[mon][icao_port]

    if icao_port == "":
        port_name = ""
    else:
        port_name = f" in {icao_port}"

    seri_hist2 = pd.Series(dict_hist2)

    months = seri_hist2.index

    # Constructing figure
    fig2 = go.Figure(data=[go.Bar(
            x=months,
            y=seri_hist2,
            textposition='outside',
            marker_color='#133E87'
        )])

    fig2.update_layout(
        title=f"{dropdown_value} Flights Per Month for {airport_code}{port_name}",
        xaxis_title="Month",
        yaxis_title=f"Number of {dropdown_value} Flights",
        yaxis=dict(showgrid=True),
        bargap=0.2,
        height=500,
        paper_bgcolor='#F3F3E0'
    )

    return fig2

if __name__ == '__main__':
    app.run_server(debug=False, port=8050)
