# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 01:03:11 2024

@author: jjoo0
"""
#%%
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import plotly.figure_factory as ff

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import scipy.ndimage as ndimage
from scipy.ndimage import binary_dilation

from datetime import datetime, timedelta

import tropycal.tracks as tracks

import itertools


pangu_dir = r'/Data/home/jjoo0727/Pangu-Weather'

lat_indices = np.linspace(90, -90, 721)
lon_indices = np.linspace(-180, 180, 1441)[:-1]

proj = ccrs.PlateCarree(central_longitude=180)
original_cmap = plt.get_cmap("BrBG")
truncated_BrBG = truncate_colormap(original_cmap, minval=0.35, maxval=1.0) #수증기 colormap 지정

pres_list = ['1000','925','850','700','600','500','400','300','250','200','150','100','50']
pres=500                                            #살펴볼 기압면 결정
p=pres_list.index(str(pres))

surface_factor = ['MSLP', 'U10', 'V10', 'T2M']
surface_dict = {'MSLP':0, 'U10':1, 'V10':2, 'T2M':3}
upper_factor = ['z', 'q', 't', 'u', 'v']
upper_dict = {'z':0, 'q':1, 't':2, 'u':3, 'v':4}

#위경도 범위 지정 함수
def latlon_extent(lon_min, lon_max, lat_min, lat_max):    
    lon_min, lon_max = lon_min-180, lon_max-180  
     
    # 위경도 범위를 데이터의 행과 열 인덱스로 변환
    lat_start = np.argmin(np.abs(lat_indices - lat_max)) 
    lat_end = np.argmin(np.abs(lat_indices - lat_min))
    lon_start = np.argmin(np.abs(lon_indices - lon_min))
    lon_end = np.argmin(np.abs(lon_indices - lon_max))
    latlon_ratio = (lon_max-lon_min)/(lat_max-lat_min)
    extent=[lon_min, lon_max, lat_min, lat_max]
    return lat_start, lat_end, lon_start, lon_end, extent, latlon_ratio

#수증기 색상 함수
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'truncated_' + cmap.name, cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def calculate_vorticity(u, v, lat_grid, lon_grid):
    # Constants
    earth_radius = 6371e3  # in meters
    deg_to_rad = np.pi / 180

    # Pre-calculate deltas for longitude and latitude
    delta_lon = 0.25 * deg_to_rad * earth_radius * np.cos(lat_grid * deg_to_rad)
    delta_lat = 0.25 * deg_to_rad * earth_radius * np.ones_like(lat_grid)


    # Initialize vorticity array with nan values
    vorticity = np.full_like(u, np.nan)

    # Calculate partial derivatives using slicing, avoiding boundaries
    dv_dx = np.empty_like(v)
    dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * delta_lon[:, 1:-1])
    dv_dx[:, 0] = dv_dx[:, -1] = np.nan

    du_dy = np.empty_like(u)
    du_dy[1:-1, :] = (u[:-2, :] - u[2:, :]) / (2 * delta_lat[1:-1, :])
    du_dy[0, :] = du_dy[-1, :] = np.nan

    # Calculate vorticity avoiding boundaries
    vorticity[1:-1, 1:-1] = dv_dx[1:-1, 1:-1] - du_dy[1:-1, 1:-1]

    return vorticity

def storm_info(storm_name, storm_year):
    ibtracs = tracks.TrackDataset(basin='all',source='ibtracs',ibtracs_mode='jtwc_neumann',catarina=True)
    storm = ibtracs.get_storm((storm_name,storm_year))
    storm.to_dataframe()
    first_index = np.where((storm['wmo_vmax']>=35) == True)[0][0]
    storm_lon = storm['lon'][first_index:]-180
    storm_lat = storm['lat'][first_index:]
    storm_mslp = storm['mslp'][first_index:]
    storm_time = storm['time'][first_index:]
    return storm_lon, storm_lat, storm_mslp, storm_time

#태풍 발생 위치 주변 & 10m/s 이상 지역 주변
def plot_min_value(ax, data, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid, 
                   wind_speed, pred_str, storm_lon, storm_lat, storm_mslp, storm_time, min_position, mask_size=10, init_size=3):
    
    pred_str = datetime.strptime(pred_str, "%Y/%m/%d/%HUTC")
    init_str = storm_time[0]
    
    
    #태풍 발생 시점 이전엔 찾기 X
    if pred_str < init_str:
        return min_position
    
    storm_lon = storm_lon[storm_time<=pred_str]
    storm_lat = storm_lat[storm_time<=pred_str]
    storm_mslp = storm_mslp[storm_time<=pred_str]
    storm_time = storm_time[storm_time<=pred_str]
 
    # 데이터의 복사본 생성 및 마스킹
    data_copy = np.copy(data)
    wind_mask = wind_speed > 10         # wind_speed > 10인 조건을 만족하는 픽셀에 대한 마스크 생성
    expanded_wind_mask = binary_dilation(wind_mask, structure=np.ones((5, 5)))  # wind_mask의 주변 2픽셀 확장
    data_copy[~expanded_wind_mask] = np.nan # 확장된 마스크를 사용하여 wind_speed > 10 조건과 그 주변 2픽셀 이외의 위치를 NaN으로 설정
    
    if len(min_position) < 1:
        data_copy[(lat_grid > (storm_lat[0]+init_size))|(lat_grid < (storm_lat[0]-init_size))] = np.nan   
        data_copy[(lon_grid > (storm_lon[0]+init_size))|(lon_grid < (storm_lon[0]-init_size))] = np.nan 
    
    
    if pred_str > init_str and min_position:
        _, _, last_min_idx, _, _ = min_position[-1]
        row_start = max(0, last_min_idx[0] - mask_size)
        row_end = min(data_copy.shape[0], last_min_idx[0] + mask_size + 1)  # +1은 Python의 슬라이싱이 상한을 포함하지 않기 때문
        col_start = max(0, last_min_idx[1] - mask_size)
        col_end = min(data_copy.shape[1], last_min_idx[1] + mask_size + 1)
        data_nan_filled = np.full(data_copy.shape, np.nan)
        data_nan_filled[row_start:row_end, col_start:col_end] = data_copy[row_start:row_end, col_start:col_end]
        data_copy = data_nan_filled

    if np.isnan(data_copy).all():
        print("모든 값이 NaN입니다. 유효한 최소값이 없습니다.")
    else:
        # 최소값 찾기
        min_index = np.unravel_index(np.nanargmin(data_copy), data_copy.shape)
        min_value = np.nanmin(data_copy)
        min_lat = lat_indices[lat_start + min_index[0]]
        min_lon = lon_indices[lon_start + min_index[1]]
        min_position.append([min_lon, min_lat, min_index, pred_str.strftime("%Y/%m/%d/%HUTC"), min_value])
        ax.text(min_lon, min_lat, f'{min_value/100:.0f}hPa', transform=ax.projection, color='red', 
                horizontalalignment='center', verticalalignment='bottom', fontsize=20, fontweight='bold')
    
    norm_p = mcolors.Normalize(vmin=950, vmax=1020)
    
    for i, (lon, lat, idx, p_str, min_pres) in enumerate(min_position):
        if p_str.endswith('00UTC'):
            dx, dy = 3, -3  # 시간 나타낼 위치 조정
            new_lon, new_lat = lon + dx, lat + dy
            
            # annotate를 사용하여 텍스트와 함께 선(화살표)을 그림
            ax.annotate(p_str[5:], xy=(lon, lat), xytext=(new_lon, new_lat),
                    textcoords='data', arrowprops=dict(arrowstyle="-", color='gray'),
                    color='red', horizontalalignment='center', verticalalignment='center', fontsize=8,
                    transform=ax.projection)


    lons = [pos[0] for pos in min_position]
    lats = [pos[1] for pos in min_position]
    min_values = [pos[4]/100 for pos in min_position]
    
    if len(min_position) > 1:
            ax.plot(lons, lats, color='red', transform=ax.projection, linestyle='-', marker='', label = 'model pred')
    
    min_p = ax.scatter(lons, lats, c=min_values, cmap='jet_r', norm=norm_p, transform=ax.projection, zorder=2)
    cbar_min_p = plt.colorbar(min_p, orientation='horizontal', fraction=0.046, pad=0.07)
    cbar_min_p.set_label('Minimum Pressure(hPa)', fontsize=16)
    ax.plot(storm_lon, storm_lat, color='black', linestyle='-', marker='o', label = 'best track', transform=ax.projection, zorder=3)
    ax.legend()
    return min_position

#ax 배경 지정
def setup_map(ax):
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    ax.coastlines()
    ocean_color = mcolors.to_rgba((233/255, 240/255, 248/255))
    land_color = mcolors.to_rgba((255/255, 255/255, 255/255))
    ax.add_feature(cfeature.OCEAN, color=ocean_color)
    ax.add_feature(cfeature.LAND, color=land_color, edgecolor='none')

#contour 함수
def weather_map_contour(ax, lon_grid, lat_grid, data):
    levels = np.arange(920, 1040, 4)
    bold_levels = np.arange(904,1033,16)
    levels = levels[~np.isin(levels, bold_levels)]
    filtered_data = ndimage.gaussian_filter(data, sigma=3, order=0)
    cs = ax.contour(lon_grid, lat_grid, filtered_data / 100, levels=levels, colors='black', transform=proj)
    ax.clabel(cs, cs.levels,inline=True, fontsize=10)
    cs_bold = ax.contour(lon_grid, lat_grid, filtered_data / 100, levels=bold_levels, colors='black', transform=proj, linewidths=3)
    ax.clabel(cs_bold, cs_bold.levels, inline=True, fontsize=10)
    

def process_meteorological_data(surface, surface_dict, upper, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid):
    
    # Extract surface variables
    mslp = surface[surface_dict['MSLP']][lat_start:lat_end + 1, lon_start:lon_end + 1]
    u10 = surface[surface_dict['U10']][lat_start:lat_end + 1, lon_start:lon_end + 1]
    v10 = surface[surface_dict['V10']][lat_start:lat_end + 1, lon_start:lon_end + 1]
    
    # Calculate geopotential heights and their difference
    z_850 = upper[upper_dict['z'], 2, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1] / 9.80665
    z_200 = upper[upper_dict['z'], 9, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1] / 9.80665
    z_diff = z_200 - z_850
    
    # Calculate 10-meter wind speed
    wind_speed = np.sqrt(u10**2 + v10**2)
    
    u_850 = upper[upper_dict['u'], 2, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1]
    v_850 = upper[upper_dict['v'], 2, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1]
    vorticity = calculate_vorticity(u_850, v_850, lat_grid, lon_grid)
    
    return mslp, wind_speed, z_diff, vorticity


#%%
#위경도 범위 지정
#동반구는 0~180, 서반구는 180~360y
lat_start, lat_end, lon_start, lon_end, extent, latlon_ratio = latlon_extent(110,160,5,45)  
lon_grid, lat_grid = np.meshgrid(lon_indices[lon_start:lon_end + 1], lat_indices[lat_start:lat_end + 1])

storm_lon, storm_lat, storm_mslp, storm_time = storm_info('maysak', 2020)   #태풍 영문명, 년도 입력
predict_interval_list = np.arange(0,24*7+1,6)[1:]     #살펴볼 예측 시간 지정
year = ['2020']
month = ['08']
day = ['26','27','28']
times = ['00','06','12','18']


for y, m, d, tm in itertools.product(year, month, day, times):
    #ERA5 initial map
    time_str = f'{y}/{m}/{d}/{tm}UTC'
    print(time_str)
    input_data_dir = rf'{pangu_dir}/input_data/{time_str}'
    
    min_position = []  # 최저값 위치를 저장할 리스트

    predict_str = time_str
    surface = np.load(os.path.join(input_data_dir, 'surface.npy')).astype(np.float32)  
    upper = np.load(os.path.join(input_data_dir, 'upper.npy')).astype(np.float32)  
    mslp, wind_speed, z_diff, vorticity = process_meteorological_data(surface, surface_dict, upper, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
    
    
    fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
    ax.set_title(f'{time_str} Surface (+0h)', fontsize=20)
    min_position = plot_min_value(ax, mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid,
                                                    wind_speed, predict_str, storm_lon, storm_lat, storm_mslp, storm_time, min_position)
    setup_map(ax)
    weather_map_contour(ax, lon_grid, lat_grid, mslp)

    contourf = ax.contourf(lon_grid, lat_grid, vorticity, cmap='seismic', levels = np.linspace(-10**-4*5,10**-4*5,21),extend='both')
    # contourf = ax.contourf(lon_grid, lat_grid, z_diff, cmap='jet', levels = np.linspace(10000,11500,16), alpha=0.5)
    # contourf = ax.contourf(lon_grid, lat_grid, wind_speed, cmap='jet', levels = np.linspace(0,30,7), alpha=0.5)
    
    cbar = plt.colorbar(contourf, ax=ax, orientation='horizontal', extend='both', fraction=0.046, pad=0.04)
    cbar.set_label('Vorticity', fontsize=16)  # You can customize the label here
    # cbar.set_label('200-850hPa Thickness(m)', fontsize=16)  # You can customize the label here
    # cbar.set_label('10m Wind Speed(m/s)', fontsize=16)  # You can customize the label here
    
    if not os.path.exists(f'{pangu_dir}/plot/typhoon/{time_str}'):
        os.makedirs(f'{pangu_dir}/plot/typhoon/{time_str}')
    plt.close(fig)
    fig.savefig(f'{pangu_dir}/plot/typhoon/{time_str}/0h_vc.png')

    #모델 예측 그리기
    for predict_interval in predict_interval_list:

        time_obj = datetime.strptime(time_str, "%Y/%m/%d/%HUTC")
        predict_time = time_obj + timedelta(hours=int(predict_interval))
        predict_str = predict_time.strftime("%Y/%m/%d/%HUTC")
        output_data_dir = rf'{pangu_dir}/output_data/{time_str}'
        
        # Load surface data
        surface = np.load(os.path.join(output_data_dir, rf'surface/{predict_interval}h.npy')).astype(np.float32)
        upper = np.load(os.path.join(output_data_dir, rf'upper/{predict_interval}h.npy')).astype(np.float32)
        mslp, wind_speed, z_diff, vorticity = process_meteorological_data(surface, surface_dict, upper, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
        
        fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
        ax.set_title(f'{time_str} (+{predict_interval}h)  predict: {predict_str}', fontsize=20)
        min_position = plot_min_value(ax, mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid,
                                                        wind_speed, predict_str, storm_lon, storm_lat, storm_mslp, storm_time, min_position)
        setup_map(ax)
        weather_map_contour(ax, lon_grid, lat_grid, mslp)

        contourf = ax.contourf(lon_grid, lat_grid, vorticity, cmap='seismic', levels = np.linspace(-10**-4*5,10**-4*5,21),extend='both')
        # contourf = ax.contourf(lon_grid, lat_grid, wind_speed, cmap='jet', levels = np.linspace(0,30,7), alpha=0.5)
        # contourf = ax.contourf(lon_grid, lat_grid, z_diff, cmap='jet', levels = np.linspace(10000,11500,16), alpha=0.5)
        cbar = plt.colorbar(contourf, ax=ax, orientation='horizontal', extend='both', fraction=0.046, pad=0.04)
        cbar.set_label('Vorticity($S^{-1}$)', fontsize=16)  # You can customize the label here
        # cbar.set_label('200-850hPa Thickness(m)', fontsize=16)  # You can customize the label here
        # cbar.set_label('10m Wind Speed(m/s)', fontsize=16)  # You can customize the label here
        plt.close(fig)
        fig.savefig(f'{pangu_dir}/plot/typhoon/{time_str}/{predict_interval}h_vc.png')