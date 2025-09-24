import streamlit as st
import requests
import folium
import pandas as pd
import numpy as np
import json
import streamlit.components.v1 as components
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import plotly.express as px
import plotly.graph_objects as go
import base64
import time

# GeoJSON data for Tasikmalaya
geojson_data = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [108.19886016846, -7.2762899398804],
                    [108.1994934082, -7.2793798446655],
                    [108.20127868652, -7.2814908027649],
                    [108.20220947266, -7.2825889587402],
                    [108.20729827881, -7.2859191894531],
                    [108.21282196045, -7.2895288467407],
                    [108.21633911133, -7.2904391288757],
                    [108.21881866455, -7.2913589477539],
                    [108.22101593018, -7.2923898696899],
                    [108.22395324707, -7.2951397895812],
                    [108.22841644287, -7.2973899841309],
                    [108.22898101807, -7.3007698059081],
                    [108.23220825195, -7.3028888702393],
                    [108.23375701904, -7.3039197921753],
                    [108.23809051514, -7.3061599731445],
                    [108.24085998535, -7.3082799911498],
                    [108.24253845215, -7.3097701072692],
                    [108.24507904053, -7.3120698928833],
                    [108.24697875977, -7.3120608329773],
                    [108.24823760986, -7.311978816986],
                    [108.24955749512, -7.3120789527892],
                    [108.25118255615, -7.3125901222229],
                    [108.25199890137, -7.3132090568542],
                    [108.25277709961, -7.314109802246],
                    [108.25318908691, -7.3150582313536],
                    [108.2537689209, -7.3166189193726],
                    [108.25414276123, -7.3176307678223],
                    [108.25634002686, -7.3192400932312],
                    [108.25778961182, -7.319529056549],
                    [108.26145935059, -7.3202590942383],
                    [108.26319122314, -7.3194589614868],
                    [108.26508331299, -7.3185791969299],
                    [108.2671585083, -7.3172588348389],
                    [108.27060699463, -7.3166689872741],
                    [108.27308654785, -7.3181099891661],
                    [108.27430725098, -7.3207488059997],
                    [108.27680206299, -7.3288488388062],
                    [108.27910614014, -7.3313188552856],
                    [108.28302764893, -7.3322901725769],
                    [108.28550720215, -7.3325209617615],
                    [108.28762817383, -7.3330402374267],
                    [108.28897094727, -7.3326001167297],
                    [108.28753662109, -7.3361811637878],
                    [108.28526306152, -7.3384599685668],
                    [108.27982330322, -7.3435611724852],
                    [108.27799987793, -7.3464188575744],
                    [108.27648162842, -7.3500580787658],
                    [108.27755737305, -7.3533501625061],
                    [108.2801361084, -7.3569188117981],
                    [108.2795791626, -7.3600301742554],
                    [108.27752685547, -7.3614292144775],
                    [108.2760925293, -7.3633089065552],
                    [108.27583312988, -7.3656091690063],
                    [108.27628326416, -7.3688588142394],
                    [108.27468109131, -7.3733792304993],
                    [108.27082824707, -7.3796691894531],
                    [108.2688369751, -7.3829011917114],
                    [108.26773071289, -7.3848791122435],
                    [108.26484680176, -7.3891887664795],
                    [108.26362609863, -7.3920488357543],
                    [108.26371002197, -7.3955001831055],
                    [108.26567840576, -7.3986291885375],
                    [108.26774597168, -7.4016990661621],
                    [108.26835632324, -7.4061298370361],
                    [108.26914215088, -7.4098582267761],
                    [108.26885986328, -7.414589881897],
                    [108.26798248291, -7.4180698394775],
                    [108.26484680176, -7.4239587783813],
                    [108.2618560791, -7.4285287857056],
                    [108.26094818115, -7.431637763977],
                    [108.2571105957, -7.4330601692199],
                    [108.25512695312, -7.4338188171386],
                    [108.25131225586, -7.4334092140197],
                    [108.24415588379, -7.4328389167786],
                    [108.23929595947, -7.4345889091492],
                    [108.23535919189, -7.4361100196838],
                    [108.23036956787, -7.4367799758911],
                    [108.22690582275, -7.4365382194519],
                    [108.22482299805, -7.4356398582458],
                    [108.21858215332, -7.4316291809081],
                    [108.21620178223, -7.4324488639831],
                    [108.21482849121, -7.4348101615906],
                    [108.21257019043, -7.4374589920043],
                    [108.20929718018, -7.4385499954224],
                    [108.20706176758, -7.4389700889587],
                    [108.20374298096, -7.4401888847351],
                    [108.20172119141, -7.4419012069702],
                    [108.19937133789, -7.4466800689697],
                    [108.19435882568, -7.4459600448608],
                    [108.19142150879, -7.4449090957641],
                    [108.18711853027, -7.4407501220703],
                    [108.18470001221, -7.4348402023315],
                    [108.18418121338, -7.4321298599243],
                    [108.18300628662, -7.4294800758361],
                    [108.1820602417, -7.422709941864],
                    [108.18077850342, -7.4180588722229],
                    [108.17980957031, -7.4121689796448],
                    [108.17961883545, -7.4095802307128],
                    [108.17945098877, -7.407259941101],
                    [108.17581176758, -7.4009289741516],
                    [108.17710876465, -7.3955001831055],
                    [108.17520904541, -7.3883800506591],
                    [108.17370605469, -7.3852190971375],
                    [108.17249298096, -7.3819298744202],
                    [108.17185974121, -7.3797898292541],
                    [108.16954803467, -7.3732810020446],
                    [108.16802215576, -7.368929862976],
                    [108.16546630859, -7.3618421554565],
                    [108.16353607178, -7.3579988479614],
                    [108.16274261475, -7.3529891967773],
                    [108.16327667236, -7.3497800827026],
                    [108.16323852539, -7.3467698097229],
                    [108.16227722168, -7.3436789512634],
                    [108.16046905518, -7.33758020401],
                    [108.15836334229, -7.3346791267395],
                    [108.1556930542, -7.3335800170898],
                    [108.15603637695, -7.3293180465698],
                    [108.15746307373, -7.3249292373657],
                    [108.15772247314, -7.3224601745605],
                    [108.15814208984, -7.3173890113831],
                    [108.15747833252, -7.3154911994934],
                    [108.15627288818, -7.3135900497437],
                    [108.15463256836, -7.311779975891],
                    [108.15294647217, -7.3097701072692],
                    [108.15161895752, -7.3076300621033],
                    [108.1501083374, -7.3054490089417],
                    [108.14833068848, -7.2999200820922],
                    [108.14746856689, -7.2976789474487],
                    [108.14628601074, -7.2927289009094],
                    [108.14811706543, -7.2918291091918],
                    [108.14994049072, -7.2934699058532],
                    [108.15273284912, -7.2929592132568],
                    [108.15571594238, -7.2918701171875],
                    [108.15815734863, -7.2918200492859],
                    [108.16107177734, -7.2918791770935],
                    [108.16426849365, -7.2912998199462],
                    [108.16371917725, -7.2881498336791],
                    [108.16323852539, -7.2843599319458],
                    [108.16649627686, -7.2831411361694],
                    [108.16504669189, -7.2798790931702],
                    [108.1644821167, -7.2739491462708],
                    [108.16787719727, -7.2737507820129],
                    [108.17112731934, -7.2739791870117],
                    [108.17566680908, -7.2744889259338],
                    [108.18274688721, -7.2751998901367],
                    [108.18828582764, -7.2739901542664],
                    [108.19191741943, -7.2756490707397],
                    [108.19439697266, -7.2757091522217],
                    [108.19886016846, -7.2762899398804]
                ]]
            },
            "properties": {
                "name": "KOTA TASIKMALAYA",
                "latitude": -7.35,
                "longitude": 108.21667
            }
        }
    ]
}

API_URL = 'https://opendata.tasikmalayakota.go.id/api/bigdata/dinas_kesehatan/jumlah_balita_stunting_berdasarkan_puskesmas_di_kota'
TASIKMALAYA_COORDINATES = (-7.3500, 108.2172)

@st.cache_resource
def create_model(input_shape, layers, optimizer='adam', dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(layers[0], activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(dropout_rate))
    for units in layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    if optimizer == 'adam':
        opt = Adam(learning_rate=0.001)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=0.001)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=0.001)
    else:
        opt = Adam(learning_rate=0.001)
    
    model.compile(optimizer=opt, loss='mse')
    return model

@st.cache_data
def train_model(model, X, y, epochs, batch_size):
    return model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

@st.cache_data
def fetch_data(url, params=None):
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()['data']
    else:
        st.error(f"Error {response.status_code}: {response.text}")
        return None

def process_data(data):
    df = pd.DataFrame(data)
    df['tahun'] = pd.to_numeric(df['tahun'])
    df['jumlah_balita_stunting'] = pd.to_numeric(df['jumlah_balita_stunting'])
    return df

def display_table(df):
    kecamatan_options = sorted(df['nama_kecamatan'].unique())
    puskesmas_options = sorted(df['puskesmas'].unique())
    year_options = sorted(df['tahun'].unique())

    # Container utama untuk tabel dan filter
    st.markdown("""
    <style>
        .table-container {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 20px;
            background-color: white;
        }
        .table-header {
            background-color: #F0E7E2;
            padding: 15px;
            border-bottom: 1px solid #ddd;
        }
        .table-footer {
            padding: 10px 15px;
            border-top: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: white;
        }
    </style>
    <div class="table-container">
        <div class="table-header">
            <div style="font-weight: 600; color: #333; margin-bottom: 10px;">Filter by</div>
    """, unsafe_allow_html=True)

    # Filter dalam 3 kolom
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_kecamatan = st.selectbox("Kecamatan", ["Semua"] + kecamatan_options, key="kecamatan_filter")
    with col2:
        selected_puskesmas = st.selectbox("Puskesmas", ["Semua"] + puskesmas_options, key="puskesmas_filter")
    with col3:
        selected_year = st.selectbox("Tahun", ["Semua"] + year_options, key="year_filter")

    # Lanjutan container
    st.markdown("</div>", unsafe_allow_html=True)

    filtered_df = df
    if selected_kecamatan != "Semua":
        filtered_df = filtered_df[filtered_df['nama_kecamatan'] == selected_kecamatan]
    if selected_puskesmas != "Semua":
        filtered_df = filtered_df[filtered_df['puskesmas'] == selected_puskesmas]
    if selected_year != "Semua":
        filtered_df = filtered_df[filtered_df['tahun'] == selected_year]

    total_rows = len(filtered_df)
    rows_per_page = 10
    total_pages = (total_rows + rows_per_page - 1) // rows_per_page

    if total_rows == 0:
        st.markdown("""
        <div style="padding: 20px; text-align: center; background-color: white;">
            <p>Tidak ada data yang sesuai dengan filter yang dipilih.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Pilih halaman di bagian atas tabel
        current_page = st.number_input("Pilih halaman:", 
                                      min_value=1, 
                                      max_value=total_pages, 
                                      step=1, 
                                      value=1,
                                      key="pagination_input")

        start_idx = (current_page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)
        
        # Ambil hanya kolom yang diperlukan
        paginated_df = filtered_df.iloc[start_idx:end_idx][['nama_kecamatan', 'puskesmas', 'jumlah_balita_stunting', 'tahun']].copy()
        
        # Tambahkan nomor urut tanpa kolom khusus
        paginated_df = paginated_df.reset_index(drop=True)
        
        paginated_df = paginated_df.rename(columns={
            'nama_kecamatan': 'Kecamatan',
            'puskesmas': 'Puskesmas',
            'jumlah_balita_stunting': 'Jumlah Balita Stunting',
            'tahun': 'Tahun'
        })
        paginated_df['Tahun'] = paginated_df['Tahun'].astype(int).astype(str)

        # CSS untuk styling tabel
        st.markdown("""
        <style>
            .dataframe-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.95rem;
                background-color: white;
            }
            .dataframe-table thead {
                background-color: #F0E7E2;
            }
            .dataframe-table th {
                color: #333;
                font-weight: 600;
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            .dataframe-table td {
                padding: 10px 15px;
                text-align: left;
                border-bottom: 1px solid #eee;
                background-color: white;
            }
            .dataframe-table tr:last-child td {
                border-bottom: none;
            }
            .dataframe-table tr:hover td {
                background-color: #f9f9f9;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Menampilkan tabel sebagai HTML tanpa index
        table_html = paginated_df.to_html(
            classes='dataframe-table', 
            index=False,  # Tidak menampilkan index/nomor
            border=0,
            justify='left'
        )
        
        # Menampilkan tabel
        st.write(table_html, unsafe_allow_html=True)
        
        # Informasi paginasi di footer tabel
        st.markdown(f"""
        <div class="table-footer">
            <div>Rows per page: 10</div>
            <div>{start_idx+1}-{end_idx} of {total_rows}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Penutup container utama
    st.markdown("</div>", unsafe_allow_html=True)

def display_average_original_data(df):
    # Calculate average stunting per year
    yearly_avg = df.groupby('tahun').agg({
        'jumlah_balita_stunting': ['mean', 'sum', 'count']
    }).reset_index()
    
    yearly_avg.columns = ['Tahun', 'Rata-rata Stunting', 'Total Stunting', 'Jumlah Puskesmas']
    
    # Format the columns
    yearly_avg['Rata-rata Stunting'] = yearly_avg['Rata-rata Stunting'].round(2)
    yearly_avg['Tahun'] = yearly_avg['Tahun'].astype(int).astype(str)
    
    # Container untuk tabel rata-rata
    st.markdown("""
    <div class="table-container">
        <div class="table-header">
            <div style="font-weight: 600; color: #333;">Rata-rata Stunting Data Asli per Tahun</div>
        </div>
    """, unsafe_allow_html=True)
    
    # CSS untuk tabel rata-rata
    st.markdown("""
        <style>
        .yearly-avg-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
            background-color: white;
        }
        .yearly-avg-table thead {
            background-color: #F0E7E2;
        }
        .yearly-avg-table th {
            color: #333;
            padding: 12px 15px;
            text-align: center;
            font-weight: 600;
            border-bottom: 1px solid #ddd;
        }
        .yearly-avg-table td {
            padding: 10px 15px;
            text-align: center;
            border-bottom: 1px solid #eee;
            background-color: white;
        }
        .yearly-avg-table tr:last-child td {
            border-bottom: none;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Convert DataFrame to HTML table
    table_html = yearly_avg.to_html(
        classes='yearly-avg-table',
        index=False,  # Tidak menampilkan index/nomor
        float_format=lambda x: '{:.2f}'.format(x) if isinstance(x, float) else x
    )
    
    # Display the table
    st.write(table_html, unsafe_allow_html=True)
    
    # Footer untuk tabel rata-rata
    st.markdown(f"""
    <div class="table-footer">
        <div>Jumlah Tahun: {len(yearly_avg)}</div>
        <div>Data Sumber: Dinas Kesehatan Kota Tasikmalaya</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Penutup container
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Grafik di bawah tabel
    fig = px.line(
        yearly_avg,
        x='Tahun',
        y='Rata-rata Stunting',
        title='Tren Rata-rata Stunting Data Asli per Tahun',
        markers=True,
        color_discrete_sequence=['#8B5A2B']  # Warna coklat tua
    )
    fig.update_traces(
        line_color='#C59F89',  # Warna coklat muda untuk garis
        marker_color='#8B5A2B',  # Warna coklat tua untuk marker
        line_width=3,
        marker_size=8
    )
    fig.update_layout(
        xaxis_title='Tahun',
        yaxis_title='Rata-rata Jumlah Stunting',
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        title_font=dict(size=18, color='#5C3D2E')
    )
    st.plotly_chart(fig)

def get_coordinates(puskesmas_name):
    geolocator = Nominatim(user_agent="geospasial_stunting_app")
    time.sleep(1)
    try:
        location = geolocator.geocode(f"{puskesmas_name}, Kota Tasikmalaya, Indonesia")
        return (location.latitude, location.longitude) if location else None
    except (GeocoderTimedOut, GeocoderUnavailable):
        return None

def create_map(data, selected_year):
    m = folium.Map(location=TASIKMALAYA_COORDINATES, zoom_start=12, tiles='CartoDB positron')
    filtered_data = [item for item in data if item['tahun'] == selected_year]
    
    # Gunakan GeoJSON dari variabel yang sudah didefinisikan
    try:
        # Tambahkan batas wilayah Kota Tasikmalaya
        folium.GeoJson(
            geojson_data,
            name='Batas Kota Tasikmalaya',
            style_function=lambda feature: {
                'color': '#000000',  # Warna garis hitam
                'weight': 2,         # Ketebalan garis
                'fillColor': '#3b82f6', # Warna isi
                'fillOpacity': 0.1    # Transparansi isi
            },
            tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Wilayah:'])
        ).add_to(m)
        
    except Exception as e:
        st.warning(f"Tidak dapat memuat data batas daerah: {str(e)}")
    
    # Tambahkan marker untuk setiap puskesmas
    for item in filtered_data:  
        puskesmas = item['puskesmas']
        jumlah_stunting = int(item['jumlah_balita_stunting'])
        coordinates = get_coordinates(puskesmas)
        
        if coordinates:
            if jumlah_stunting < 100:
                color = 'green'
                icon = 'check-circle'
            elif jumlah_stunting <= 249:
                color = 'orange'
                icon = 'exclamation-triangle'
            else:
                color = 'red'
                icon = 'exclamation-circle'
                
            folium.Marker(
                location=coordinates,
                popup=f"<strong>Puskesmas:</strong> {puskesmas}<br><strong>Jumlah Balita Stunting:</strong> {jumlah_stunting}<br><strong>Tahun:</strong> {selected_year}",
                tooltip=puskesmas,
                icon=folium.Icon(color=color, icon=icon, prefix='fa')
            ).add_to(m)

    # Tambahkan layer tambahan
    folium.TileLayer(
        'Stamen Terrain',
        name='Stamen Terrain',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.'
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    return m

def predict_stunting(df, years_to_predict, layers, epochs, optimizer, dropout_rate, batch_size):
    last_year = df['tahun'].max()
    future_years = range(last_year + 1, last_year + years_to_predict + 1)
    
    predictions = []
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    for kecamatan in df['nama_kecamatan'].unique():
        for puskesmas in df[df['nama_kecamatan'] == kecamatan]['puskesmas'].unique():
            puskesmas_data = df[(df['nama_kecamatan'] == kecamatan) & (df['puskesmas'] == puskesmas)]
            
            if len(puskesmas_data) > 1:
                X = puskesmas_data['tahun'].values.reshape(-1, 1)
                y = puskesmas_data['jumlah_balita_stunting'].values.reshape(-1, 1)
                
                X_scaled = scaler_X.fit_transform(X)
                y_scaled = scaler_y.fit_transform(y)
                
                model = create_model(1, layers, optimizer, dropout_rate)
                history = train_model(model, X_scaled, y_scaled, epochs, batch_size)
                
                y_pred_scaled = model.predict(X_scaled)
                y_pred = scaler_y.inverse_transform(y_pred_scaled)
                mae = mean_absolute_error(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                accuracy = 1 - (mae / np.mean(y))
                
                future_X = np.array(future_years).reshape(-1, 1)
                future_X_scaled = scaler_X.transform(future_X)
                future_y_scaled = model.predict(future_X_scaled)
                future_y = scaler_y.inverse_transform(future_y_scaled)
                
                for year, prediction in zip(future_years, future_y):
                    predictions.append({
                        'nama_kecamatan': kecamatan,
                        'puskesmas': puskesmas,
                        'tahun': year,
                        'jumlah_balita_stunting': max(0, int(prediction[0])),  # Ensure non-negative predictions
                        'mae': mae,
                        'rmse': rmse,
                        'accuracy': accuracy
                    })
            else:
                for year in future_years:
                    predictions.append({
                        'nama_kecamatan': kecamatan,
                        'puskesmas': puskesmas,
                        'tahun': year,
                        'jumlah_balita_stunting': puskesmas_data['jumlah_balita_stunting'].values[0],
                        'mae': 0,
                        'rmse': 0,
                        'accuracy': 1
                    })
    
    return pd.DataFrame(predictions), history

def display_prediction_table(df):
    total_rows = len(df)
    rows_per_page = 10
    total_pages = (total_rows + rows_per_page - 1) // rows_per_page

    if total_rows == 0:
        st.write("Tidak ada data prediksi yang tersedia.")
    else:
        current_page = st.number_input("Pilih halaman:", min_value=1, max_value=total_pages, step=1, value=1)

        start_idx = (current_page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)
        
        df = df.reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = 'Nomor'
        
        paginated_df = df.iloc[start_idx:end_idx]
        paginated_df = paginated_df.rename(columns={
            'nama_kecamatan': 'Kecamatan',
            'puskesmas': 'Puskesmas',
            'jumlah_balita_stunting': 'Jumlah Balita Stunting',
            'tahun': 'Tahun',
            'accuracy': 'Akurasi'
        })

        paginated_df['Tahun'] = paginated_df['Tahun'].astype(int).astype(str)
        paginated_df['Akurasi'] = paginated_df['Akurasi'].apply(lambda x: f"{x:.2%}")

        st.write(f"Menampilkan halaman {current_page} dari {total_pages}, total data: {total_rows}")
        
        # Convert the dataframe to HTML, without showing the index column to avoid duplicate headers
        table_html = paginated_df[['Kecamatan', 'Puskesmas', 'Jumlah Balita Stunting', 'Tahun', 'Akurasi']].to_html(classes='table', index=False)
        
        # Display the HTML table
        st.markdown(table_html, unsafe_allow_html=True)

        st.write(f'Halaman {current_page} dari {total_pages}')

def display_average_prediction_table(df):
    # Calculate average stunting per year
    yearly_avg = df.groupby('tahun').agg({
        'jumlah_balita_stunting': ['mean', 'sum', 'count'],
        'accuracy': 'mean'
    }).reset_index()
    
    yearly_avg.columns = ['Tahun', 'Rata-rata Stunting', 'Total Stunting', 'Jumlah Puskesmas', 'Rata-rata Akurasi']
    
    # Format the columns
    yearly_avg['Rata-rata Stunting'] = yearly_avg['Rata-rata Stunting'].round(2)
    yearly_avg['Rata-rata Akurasi'] = yearly_avg['Rata-rata Akurasi'].apply(lambda x: f"{x:.2%}")
    yearly_avg['Tahun'] = yearly_avg['Tahun'].astype(int).astype(str)
    
    # Display the table with styling
    st.markdown("""
        <style>
        .yearly-avg-table {
            width: 100%;
            margin: 1em 0;
            border-collapse: collapse;
        }
        .yearly-avg-table th {
            background-color: #1e3a8a;
            color: white;
            padding: 12px;
            text-align: center !important;
            font-weight: bold;
        }
        .yearly-avg-table td {
            padding: 10px;
            text-align: center !important;
            border: 1px solid #ddd;
        }
        .yearly-avg-table tr:nth-child(even) {
            background-color: #f8fafc;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.subheader("📊 Prediksi Rata-rata Stunting per Tahun")
    
    # Convert DataFrame to HTML table
    table_html = yearly_avg.to_html(
        classes='yearly-avg-table',
        index=False,
        float_format=lambda x: '{:.2f}'.format(x) if isinstance(x, float) else x
    )
    
    # Display the table
    st.markdown(table_html, unsafe_allow_html=True)
    
    # Create visualization
    fig = px.line(
        yearly_avg,
        x='Tahun',
        y='Rata-rata Stunting',
        title='Tren Prediksi Rata-rata Stunting per Tahun',
        markers=True,
        color_discrete_sequence=['#8B5A2B']  # Warna coklat tua
    )
    fig.update_traces(
        line_color='#C59F89',  # Warna coklat muda untuk garis
        marker_color='#8B5A2B',  # Warna coklat tua untuk marker
        line_width=3,
        marker_size=8,
        line_dash='dash'  # Garis putus-putus untuk prediksi
    )
    fig.update_layout(
        xaxis_title='Tahun',
        yaxis_title='Rata-rata Jumlah Stunting',
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        title_font=dict(size=18, color='#5C3D2E')
    )
    st.plotly_chart(fig)

def display_prediction_chart(df):
    aggregated_df = df.groupby(['nama_kecamatan', 'tahun']).agg({
        'jumlah_balita_stunting': 'sum'
    }).reset_index()

    df['detail_puskesmas'] = df.apply(
        lambda x: f"{x['puskesmas']} ({x['jumlah_balita_stunting']})", axis=1
    )

    puskesmas_details = df.groupby(['nama_kecamatan', 'tahun'])['detail_puskesmas'].apply(lambda x: ', '.join(x)).reset_index()
    
    merged_df = pd.merge(aggregated_df, puskesmas_details, on=['nama_kecamatan', 'tahun'])

    merged_df['tahun'] = merged_df['tahun'].astype(int)

    fig = px.scatter(
        merged_df,
        x='tahun',
        y='jumlah_balita_stunting',
        color='nama_kecamatan',
        title='Prediksi Jumlah Balita Stunting per Kecamatan di Kota Tasikmalaya',
        labels={
            'jumlah_balita_stunting': 'Jumlah Balita Stunting',
            'tahun': 'Tahun',
            'nama_kecamatan': 'Kecamatan'
        },
        hover_data={'detail_puskesmas': True},
        color_discrete_sequence=[
            '#8B5A2B', '#C59F89', '#A67C52', '#DB9057', '#5C3D2E',
            '#8C6E46', '#B89A76', '#7A5C3C', '#9D7E5B', '#6B4F32'
        ]  # Palet warna coklat
    )
    
    fig.update_traces(
        marker=dict(size=15, line=dict(width=1, color='DarkSlateGrey')),
        selector=dict(mode='markers')
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend_title_text='Kecamatan',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        font=dict(size=14),
        title_font=dict(size=18, color='#5C3D2E')
    )
    
    fig.update_xaxes(
        tickmode='linear',
        dtick=1,
        tickformat='d'  
    )
    
    st.plotly_chart(fig)

# ================================
# TAMPILAN UTAMA APLIKASI
# ================================
st.set_page_config(page_title="Geospasial Balita Stunting", page_icon="📊", layout="wide")

# Custom CSS dengan warna background baru
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    * {
        font-family: 'Roboto', sans-serif !important;
    }

    body {
        background: linear-gradient(135deg, #F7F2EF 41%, #C59F89 100%) !important;
        background-attachment: fixed;
        color: #1e293b;
    }

    .stApp {
        background: linear-gradient(135deg, #F7F2EF 41%, #C59F89 100%) !important;
        background-attachment: fixed;
    }

    .hero-section {
        background: linear-gradient(135deg, #8B5A2B 0%, #C59F89 100%);
        padding: 4rem 2rem;
        border-radius: 16px;
        margin-bottom: 3rem;
        text-align: center;
        color: white;
    }
    
    .hero-section h1 {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: white;
    }
    
    .hero-section h2 {
        font-size: 1.8rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
        color: white;
    }
    
    .hero-section p {
        font-size: 1.2rem;
        max-width: 800px;
        margin: 0 auto 2rem;
        color: #f0f0f0;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        border: 2px solid #DB9057;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .metric-label {
        font-size: 16px;
        color: #DB9057 !important;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 34px;
        font-weight: 700;
        color: #DB9057;
        margin-bottom: 8px;
        line-height: 1.2;
    }
    
    .metric-card div:last-child {
        font-size: 15px;
        color: #555;
        padding: 0 10px;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin-top: 30px;
    }

    .feature-card {
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
        background: linear-gradient(
            135deg,
            #C4A08D 0%,
            #FFFFFF 38%,
            #F7F2EF 69%,
            #D8C0B4 100%
        );
        transition: transform 0.3s ease;
        border: 1px solid #E0D0C5;
    }

    .feature-card:hover {
        transform: translateY(-7px);
        box-shadow: 0 8px 22px rgba(0,0,0,0.15);
    }

    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 15px;
    }   

    .feature-card h3 {
        color: #5C3D2E;
        margin-top: 0;
        margin-bottom: 15px;
        font-size: 1.4rem;
    }

    .feature-card p {
        color: #5C5047;
        line-height: 1.6;
        margin-bottom: 0;
    }
    
    .section {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    .section-title {
        color: #DB9057;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.5rem;
        margin-top: 0;
    }
    
    .stButton>button {
       background: linear-gradient(90deg, #DB9057 0%, #C4A08D 100%);
        color: white;
        font-size: 1.1rem;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: 0.3s ease-in-out;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
          opacity: 0.95;
        transform: scale(1.01);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.95rem;
        background-color: white;
        border-radius: 16px;
        margin-top: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    .map-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        margin-top: 1rem;
    }
    
    .legend-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 1.5rem 0;
        flex-wrap: wrap;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 50%;
    }
    
    .legend-green {
        background-color: #10b981;
    }
    
    .legend-orange {
        background-color: #f59e0b;
    }
    
    .legend-red {
        background-color: #ef4444;
    }
    
    .search-box {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .search-box input {
        flex-grow: 1;
        padding: 0.8rem;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        font-size: 1rem;
    }
    
    .search-box button {
        background: #3b82f6;
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        cursor: pointer;
        font-weight: 500;
        transition: background 0.3s ease;
    }
    
    .search-box button:hover {
        background: #2563eb;
    }
    
    .parameter-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    
    table {
        width: 100% !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    }
    
    thead th {
        background-color: #F0E7E2 !important;
        color: #E0A67A !important;
        text-align: center !important;
        padding: 14px !important;
        font-weight: 500 !important;
        font-size: 1.1rem;
    }
    
    tbody td {
        padding: 12px !important;
        border-bottom: 1px solid #e2e8f0 !important;
        font-size: 1rem;
    }
    
    tbody tr:last-child td {
        border-bottom: none !important;
    }
    
    tbody tr:hover {
        background-color: #f0f9ff !important;
    }
    
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 20px;
        background-color: #ffffff;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .logo {
        font-weight: 700;
        font-size: 1.4rem;
        color: #333;
    }
    
    .menu {
        display: flex;
        gap: 10px;
    }
    
    .menu a {
        color: #333;
        text-decoration: none;
        font-weight: 500;
        padding: 8px 15px;
        border-radius: 4px;
        transition: background 0.2s;
    }
    
    .menu a:hover {
        background-color: #f8f9fa;
    }

      .stSelectbox [class*="material-icons"] {
        display: none !important;
    }
    
    /* Ganti dengan ikon Unicode yang sederhana */
    .stSelectbox div[data-baseweb="select"] > div:after {
        content: "▼" !important;
        color: #666 !important;
        font-size: 12px !important;
        position: absolute !important;
        right: 12px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        pointer-events: none !important;
        font-family: 'Roboto', sans-serif !important;
    }
    
    /* Pastikan select box memiliki padding yang cukup */
    .stSelectbox select {
        padding-right: 30px !important;
        appearance: none !important;
        -webkit-appearance: none !important;
        -moz-appearance: none !important;
    }
    
    /* Perbaikan untuk expander */
    .streamlit-expanderHeader::-webkit-details-marker {
        display: none !important;
    }
    
    .streamlit-expanderHeader:after {
        content: "▼" !important;
        color: #666 !important;
        font-size: 12px !important;
        margin-left: 8px !important;
        font-family: 'Roboto', sans-serif !important;
    }
    
    .streamlit-expanderHeader[aria-expanded="false"]:after {
        content: "►" !important;
    }
</style>
""", unsafe_allow_html=True)

# Navbar
st.markdown("""
<div class="nav-container">
    <div class="logo">
        <span style="color: #DD9762;">GEO</span>Stunting
    </div>
    <div class="menu">
        <a href="#beranda">Home</a>
        <a href="#peta">Peta</a>
        <a href="#tabel">Data</a>
        <a href="#prediksi">Prediksi</a>
    </div>
</div>

""", unsafe_allow_html=True)

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Load gambar lokal
image_base64 = get_base64_image("Group 189.png")  # Pastikan file ada di direktori yang sama

# HTML dengan gambar dan tanpa outline
st.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, #8B5A2B 0%, #C59F89 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 3rem;
        color: white;
    ">
        <div style="
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            max-width: 1200px;
            margin: 0 auto;
        ">
            <div style="
                flex: 1 1 500px; 
                padding: 20px;
                text-align: left;
            ">
                <h1 style="
                    font-size: 2.8rem;
                    margin-bottom: 0.8rem;
                    line-height: 1.2;
                    font-weight: 700;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    color: white;
                ">
                    Pantau Kondisi & Prediksi Stunting<br>
                    <span style="font-size: 2.4rem; font-weight: 600;">di Kota Tasikmalaya</span><br>
                    <span style="font-size: 2.0rem; font-weight: 500;">Secara Interaktif</span>
                </h1>
                <p style="
                    font-size: 1.3rem;
                    margin: 2rem 0;
                    line-height: 1.6;
                    max-width: 90%;
                ">
                    Gunakan data geospasial, tabel interaktif, dan prediksi AI untuk memahami 
                    dan mencegah stunting lebih efektif.
                </p>
            </div>
            <div style="flex: 1 1 400px; padding: 20px; text-align: center;">
                <img src="data:image/png;base64,{image_base64}" 
                     alt="Ilustrasi Stunting" 
                     style="
                         max-width: 100%;
                         height: auto;
                         border-radius: 0px;
                         box-shadow: none;
                         border: none;
                     ">
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Metrics section
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">27<span style="color: black; font-weight: bold !important;">+</span></div>
        <div>Di seluruh Kota Tasikmalaya</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">2020<span style="color: black; font-weight: bold !important;">-</span>2023</div>
        <div>Data historis yang tersedia</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">92<span style="color: black; font-weight: bold !important;">%</span></div>
        <div>Rata-rata akurasi prediksi model Kecerdasan Buatan</div>
    </div>
    """, unsafe_allow_html=True)

# Fitur sistem
st.markdown(""" 
<div class="center-container">
    <div class="center-title">
        <h2>Apa Saja Kelebihan Sistem Ini?</h2>
    </div>
""", unsafe_allow_html=True)

# Fungsi untuk encode gambar ke base64
def img_to_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load gambar PNG
icon1 = img_to_base64("Rectangle 96.png")
icon2 = img_to_base64("Rectangle 96 (1).png")
icon3 = img_to_base64("Rectangle 96 (2).png")
icon4 = img_to_base64("Rectangle 96 (3).png")

# Render markdown dengan HTML dan gambar
st.markdown(f"""
<style>
    .feature-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }}
    .feature-card {{
        background: #fff;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s ease;
    }}
    .feature-card:hover {{
        transform: scale(1.02);
    }}
    .feature-icon img {{
        width: 120px;
        height: 120px;
        margin-bottom: 1rem;
    }}
    .feature-card h3 {{
        margin: 0.8rem 0 0.5rem;
        font-size: 1.3rem;
        font-weight: 600;
    }}
    .feature-card p {{
        font-size: 1rem;
        color: #444;
    }}
</style>

<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-icon">
            <img src="data:image/png;base64,{icon1}" alt="Peta Geospasial">
        </div>
        <h3>Peta Geospasial Interaktif</h3>
        <p>Jelajahi persebaran kasus stunting secara visual berdasarkan wilayah. Peta interaktif memudahkan Anda memahami kondisi di setiap kecamatan.</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">
            <img src="data:image/png;base64,{icon2}" alt="Data Otentik">
        </div>
        <h3>Data Otentik & Terstruktur</h3>
        <p>Akses data asli jumlah stunting per tahun dalam format tabel dan grafik yang mudah dibaca. Semua data ditampilkan secara transparan dan bisa difilter sesuai kebutuhan.</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">
            <img src="data:image/png;base64,{icon3}" alt="Prediksi AI">
        </div>
        <h3>Prediksi AI yang Disesuaikan</h3>
        <p>Gunakan model kecerdasan buatan untuk memproyeksikan jumlah stunting di masa depan. Sesuaikan parameter model dan lihat hasilnya secara instan.</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">
            <img src="data:image/png;base64,{icon4}" alt="Dashboard">
        </div>
        <h3>Satu Dashboard, Semua Fitur</h3>
        <p>Semua fungsi peta, data, grafik, dan prediksi terpadu dalam satu tampilan. Tidak perlu berpindah halaman, cukup fokus pada analisis dan pengambilan keputusan.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Map section
st.markdown('<div class="section" id="peta"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Peta Stunting</h2>', unsafe_allow_html=True)
st.markdown('<p>Lihat persebaran kasus stunting balita di Kota Tasikmalaya melalui peta interaktif ini. Setiap wilayah divisualisasikan menggunakan warna yang menunjukkan tingkat kasus (rendah, sedang, tinggi), sehingga Anda dapat memahami area dengan risiko tinggi secara cepat dan mudah.</p>', unsafe_allow_html=True)

# Search box
st.markdown("""
<div class="search-box">
    <input type="text" placeholder="Cari Puskesmas, Kecamatan">
    <button>Cari</button>
</div>
""", unsafe_allow_html=True)

# Fetch and process data
data = fetch_data(API_URL)
if data:
    df = process_data(data)
    years = sorted(df['tahun'].unique())
    
    with st.expander("Filter Peta", expanded=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_year_map = st.selectbox("Pilih Tahun", 
                                             options=years,
                                             key="map_year_filter")
    
    # Tambahkan legenda dengan ikon yang sesuai
    st.markdown("""
    <div class="legend-container">
        <div class="legend-item">
            <div class="legend-color legend-green"></div>
            <span>Rendah</span>
        </div>
        <div class="legend-item">
            <div class="legend-color legend-orange"></div>
            <span>Sedang</span>
        </div>
        <div class="legend-item">
            <div class="legend-color legend-red"></div>
            <span>Tinggi</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    m = create_map(data, selected_year_map)
    m_html = m._repr_html_()
    
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    components.html(
        f"""
        <div style="width: 100%; height: 100%;">
            {m_html}
        </div>
        """, 
        height=600 
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Table section
st.markdown('<div class="section" id="tabel"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Data Stunting Tasikmalaya</h2>', unsafe_allow_html=True)
st.markdown('<p>Tabel ini menampilkan rincian jumlah kasus stunting di setiap puskesmas berdasarkan tahun pelaporan</p>', unsafe_allow_html=True)

if 'df' not in locals():
    data = fetch_data(API_URL)
    if data:
        df = process_data(data)

if 'df' in locals():
    display_table(df)
    
    # Bagian Rata-rata Stunting
    st.markdown('<div class="section"></div>', unsafe_allow_html=True)
    st.markdown('<h3>Rata-rata Stunting Data Asli per Tahun</h3>', unsafe_allow_html=True)
    st.markdown('<p>Bagian ini menampilkan rata-rata jumlah balita stunting yang tercatat setiap tahun di seluruh kecamatan Kota Tasikmalaya. Data ini diambil langsung dari laporan puskesmas tanpa prediksi atau estimasi, sehingga dapat digunakan sebagai acuan kondisi riil di lapangan.</p>', unsafe_allow_html=True)
    
    display_average_original_data(df) 

    # Bagian Grafik Tren
    st.markdown('<div class="section"></div>', unsafe_allow_html=True)
    st.markdown('<h3>Stunting dari Tahun ke Tahun</h3>', unsafe_allow_html=True)
    st.markdown('<p>Bagian ini menampilkan grafik rata-rata jumlah balita stunting di Kota Tasikmalaya berdasarkan data asli yang tercatat setiap tahun. Visualisasi ini membantu Anda memahami tren perkembangan kasus dari waktu ke waktu secara lebih jelas dan mudah dipahami.</p>', unsafe_allow_html=True)
    
    # Buat grafik tren
    yearly_avg = df.groupby('tahun').agg({
        'jumlah_balita_stunting': 'mean'
    }).reset_index()
    
    fig = px.line(
        yearly_avg,
        x='tahun',
        y='jumlah_balita_stunting',
        title='Tren Kasus Stunting di Kota Tasikmalaya',
        markers=True,
        color_discrete_sequence=['#8B5A2B']  # Warna coklat tua
    )
    fig.update_traces(
        line_color='#C59F89',  # Warna coklat muda untuk garis
        marker_color='#8B5A2B',  # Warna coklat tua untuk marker
        line_width=3,
        marker_size=8
    )
    fig.update_layout(
        xaxis_title='Tahun',
        yaxis_title='Rata-rata Jumlah Stunting',
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        title_font=dict(size=18, color='#5C3D2E')
    )
    st.plotly_chart(fig)

# Prediksi section
st.markdown('<div class="section" id="prediksi"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Perkiraan Kasus Stunting Tahun Mendatang</h2>', unsafe_allow_html=True)
st.markdown('<p>Lihat proyeksi jumlah kasus stunting untuk tahun-tahun mendatang berdasarkan data historis dan analisis AI</p>', unsafe_allow_html=True)

st.markdown("""
<div class="parameter-card">
    <h3>Parameter Model Deep Learning</h3>
    <p>Atur parameter model neural network untuk prediksi stunting</p>
</div>
""", unsafe_allow_html=True)

if 'df' in locals():
    with st.expander("Konfigurasi Model AI", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            years_to_predict = st.selectbox("Jumlah Tahun Prediksi", range(1, 6), key="years_pred")
            optimizer = st.selectbox("Optimizer", ['adam', 'rmsprop', 'sgd'], key="optimizer")
        with col2:
            num_layers = st.slider("Jumlah Layer", min_value=1, max_value=5, value=2, key="num_layers")
            epochs = st.number_input("Jumlah Epochs", min_value=1, value=200, key="epochs")
        with col3:
            dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.1, key="dropout")
            batch_size = st.number_input("Batch Size", min_value=1, value=32, key="batch_size")

        layers = [st.number_input(f"Neuron pada Layer {i+1}", min_value=1, value=32) for i in range(num_layers)]

    if st.button("Latih Model dan Prediksi", use_container_width=True, key="train_button"):
        with st.spinner('Melatih model dan membuat prediksi...'):
            predictions_df, history = predict_stunting(df, years_to_predict, layers, epochs, optimizer, dropout_rate, batch_size)
            st.session_state.predictions_df = predictions_df
            st.session_state.training_history = history
        st.success("Model telah dilatih dan prediksi telah dibuat!")

    if 'predictions_df' in st.session_state:
        st.subheader("Hasil Prediksi")
        display_prediction_table(st.session_state.predictions_df)

        st.subheader("Visualisasi Prediksi")
        display_prediction_chart(st.session_state.predictions_df)
        
        st.subheader("Rangkuman Prediksi")
        display_average_prediction_table(st.session_state.predictions_df)

# Footer
st.markdown("""
<div class="footer">
    <p style="font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem;">© 2024 Sistem Analisis Geospasial Stunting</p>
    <p>Dinas Kesehatan Kota Tasikmalaya | Data Sumber: Open Data Kota Tasikmalaya</p>
    <p style="margin-top: 1rem; font-size: 0.9rem; color: #94a3b8;">Versi 1.0 | Terakhir diperbarui: Juni 2024</p>
</div>
""", unsafe_allow_html=True)





