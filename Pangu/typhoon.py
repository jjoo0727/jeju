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
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import plotly.figure_factory as ff
import scipy.ndimage as ndimage
from scipy.ndimage import binary_dilation
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
import itertools


    # 여기에 원하는 작업 수행

pangu_dir = r'/Data/home/jjoo0727/Pangu-Weather'

lat_indices = np.linspace(90, -90, 721)
lon_indices = np.linspace(-180, 180, 1441)[:-1]

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
    delta_lat = 0.25 * deg_to_rad * earth_radius

    # Calculate partial derivatives using central differences
    dv_dx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * delta_lon)
    du_dy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * delta_lat)

    # Calculate vorticity
    vorticity = dv_dx - du_dy

    return vorticity


def plot_min_value(ax, data, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid, 
                   wind_speed, pred_str, init_str, init_pos, min_position, mask_size=10, init_size=3):
    
    pred_str = datetime.strptime(pred_str, "%Y.%m.%d %HUTC")
    init_str = datetime.strptime(init_str, "%Y.%m.%d %HUTC")
    
    if pred_str < init_str:
        return np.nan, np.nan, min_position
    

    # 데이터의 복사본 생성
    data_copy = np.copy(data)
    wind_mask = wind_speed > 10         # wind_speed > 10인 조건을 만족하는 픽셀에 대한 마스크 생성
    expanded_wind_mask = binary_dilation(wind_mask, structure=np.ones((3, 3)))  # wind_mask의 주변 1픽셀 확장
    data_copy[~expanded_wind_mask] = np.nan # 확장된 마스크를 사용하여 wind_speed > 10 조건과 그 주변 1픽셀 이외의 위치를 NaN으로 설정
    
    if len(min_position) < 1:
        data_copy[(lat_grid > (init_pos[0]+init_size))|(lat_grid < (init_pos[0]-init_size))] = np.nan   # 복사본에서 위도가 35도 초과하는 값을 NaN으로 설정
        data_copy[(lon_grid > (init_pos[1]+init_size-180))|(lon_grid < (init_pos[1]-init_size-180))] = np.nan   # 복사본에서 위도가 35도 초과하는 값을 NaN으로 설정
    
    
    if pred_str > init_str and min_position:
        last_min_lon, last_min_lat, last_min_idx, _, _ = min_position[-1]
        row_start = max(0, last_min_idx[0] - mask_size)
        row_end = min(data_copy.shape[0], last_min_idx[0] + mask_size + 1)  # +1은 Python의 슬라이싱이 상한을 포함하지 않기 때문
        col_start = max(0, last_min_idx[1] - mask_size)
        col_end = min(data_copy.shape[1], last_min_idx[1] + mask_size + 1)
        data_nan_filled = np.full(data_copy.shape, np.nan)
        data_nan_filled[row_start:row_end, col_start:col_end] = data_copy[row_start:row_end, col_start:col_end]
        data_copy = data_nan_filled

    if np.isnan(data_copy).all():
        print("모든 값이 NaN입니다. 유효한 최소값이 없습니다.")
        return np.nan, np.nan, min_position

    # 최소값 찾기
    min_index = np.unravel_index(np.nanargmin(data_copy), data_copy.shape)
    min_value = np.nanmin(data_copy)
    min_lat = lat_indices[lat_start + min_index[0]]
    min_lon = lon_indices[lon_start + min_index[1]]


    ax.text(min_lon, min_lat, f'{min_value/100:.0f}hPa', transform=ax.projection, color='red', 
            horizontalalignment='center', verticalalignment='bottom', fontsize=20, fontweight='bold')
    
    norm_p = mcolors.Normalize(vmin=950, vmax=1020)
    min_position.append([min_lon, min_lat, min_index, pred_str.strftime("%Y.%m.%d %HUTC"), min_value])
    
    for i, (lon, lat, idx, p_str, min_pres) in enumerate(min_position):
        # 선으로 연결
        if len(min_position) > 1:
            lons, lats, _, p_strs, _ = zip(*min_position)
            ax.plot(lons, lats, color='red', transform=proj, linestyle='-', marker='')

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
    print(min_values)
    
    min_p = ax.scatter(lons, lats, c=min_values, cmap='jet_r', norm=norm_p, transform=proj, zorder=2)
    cbar_min_p = plt.colorbar(min_p, orientation='horizontal', fraction=0.046, pad=0.07)
    cbar_min_p.set_label('Minimum Pressure(hPa)', fontsize=16)
    return min_lon, min_lat, min_position


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

def weather_map_contour(ax, lon_grid, lat_grid, data):
    levels = np.arange(920, 1040, 4)
    bold_levels = np.arange(904,1033,16)
    levels = levels[~np.isin(levels, bold_levels)]
    filtered_data = ndimage.gaussian_filter(data, sigma=3, order=0)
    cs = ax.contour(lon_grid, lat_grid, filtered_data / 100, levels=levels, colors='black', transform=proj)
    ax.clabel(cs, cs.levels,inline=True, fontsize=10)
    cs_bold = ax.contour(lon_grid, lat_grid, filtered_data / 100, levels=bold_levels, colors='black', transform=proj, linewidths=3)
    ax.clabel(cs_bold, cs_bold.levels, inline=True, fontsize=10)
#%%

proj = ccrs.PlateCarree(central_longitude=180)
original_cmap = plt.get_cmap("BrBG")
truncated_BrBG = truncate_colormap(original_cmap, minval=0.35, maxval=1.0) #수증기 colormap 지정

proj = ccrs.PlateCarree(central_longitude=180)  # Set central_longitude to 180
#위경도 범위 지정
#동반구는 0~180, 서반구는 180~360y
lat_start, lat_end, lon_start, lon_end, extent, latlon_ratio = latlon_extent(110,160,5,45)  

pres_list = ['1000','925','850','700','600','500','400','300','250','200','150','100','50']
pres=500                                            #살펴볼 기압면 결정
p=pres_list.index(str(pres))

surface_factor = ['MSLP', 'U10', 'V10', 'T2M']
surface_dict = {'MSLP':0, 'U10':1, 'V10':2, 'T2M':3}
upper_factor = ['z', 'q', 't', 'u', 'v']
upper_dict = {'z':0, 'q':1, 't':2, 'u':3, 'v':4}

predict_interval_list = np.arange(0,24*7+1,6)[1:]   #볼 예측 시간 지정
time_str = '2022.08.26 00UTC'                       #time_str 지정
init_str = '2022.08.28 00UTC'                       #태풍 시점
init_pos = [25.80, 149.30]                          #태풍 첫 위경도
input_data_dir = rf'{pangu_dir}/input_data/{time_str[0:4]}/{time_str[5:7]}/{time_str[8:10]}/{time_str[11:13]}UTC'


# 위경도 범위 설정
lat_start, lat_end, lon_start, lon_end, extent, latlon_ratio = latlon_extent(110, 160, 5, 45)
lon_grid, lat_grid = np.meshgrid(lon_indices[lon_start:lon_end + 1], lat_indices[lat_start:lat_end + 1])
min_position = []  # 최저값 위치를 저장할 리스트



predict_str = time_str
surface = np.load(os.path.join(input_data_dir, 'surface.npy')).astype(np.float32)  
upper = np.load(os.path.join(input_data_dir, 'upper.npy')).astype(np.float32)  
fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})

mslp = surface[surface_dict['MSLP']][lat_start:lat_end + 1, lon_start:lon_end + 1]
u10 = surface[surface_dict['U10']][lat_start:lat_end + 1, lon_start:lon_end + 1]
v10 = surface[surface_dict['V10']][lat_start:lat_end + 1, lon_start:lon_end + 1]
z_850 = upper[upper_dict['z'],2,:,:][lat_start:lat_end + 1, lon_start:lon_end + 1]/9.80665
z_200 = upper[upper_dict['z'],9,:,:][lat_start:lat_end + 1, lon_start:lon_end + 1]/9.80665
z_diff = z_200-z_850
wind_speed = np.sqrt(u10**2 + v10**2)
ax.set_title(f'{time_str} Surface (+0h)', fontsize=20)


min_lon, min_lat, min_position = plot_min_value(ax, mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid,
                                                wind_speed, predict_str, init_str, init_pos, min_position)
setup_map(ax)
weather_map_contour(ax, lon_grid, lat_grid, mslp)

# contourf = ax.contourf(lon_grid, lat_grid, vorticity, cmap='seismic', levels = np.linspace(-10**-4,10**-4,21),extend='both')
contourf = ax.contourf(lon_grid, lat_grid, z_diff, cmap='jet', levels = np.linspace(10000,11500,16), alpha=0.5)
# contourf = ax.contourf(lon_grid, lat_grid, wind_speed, cmap='jet', levels = np.linspace(0,30,7), alpha=0.5)
cbar = plt.colorbar(contourf, ax=ax, orientation='horizontal', extend='both', fraction=0.046, pad=0.04)
# cbar.set_label('Vorticity', fontsize=16)  # You can customize the label here
cbar.set_label('200-850hPa Thickness(m)', fontsize=16)  # You can customize the label here
# cbar.set_label('10m Wind Speed(m/s)', fontsize=16)  # You can customize the label here


for predict_interval in predict_interval_list:

    time_obj = datetime.strptime(time_str, "%Y.%m.%d %HUTC")
    predict_time = time_obj + timedelta(hours=int(predict_interval))
    predict_str = predict_time.strftime("%Y.%m.%d %HUTC")
    fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
    output_data_dir = rf'{pangu_dir}/output_data/{time_str[0:4]}/{time_str[5:7]}/{time_str[8:10]}/{time_str[11:]}'
    # output_data_dir = rf'{pangu_dir}/output_data/{time_str}'
    
    # Load surface data
    surface = np.load(os.path.join(output_data_dir, rf'surface/{predict_interval}h.npy')).astype(np.float32)
    upper = np.load(os.path.join(output_data_dir, rf'upper/{predict_interval}h.npy')).astype(np.float32)
    mslp = surface[surface_dict['MSLP']][lat_start:lat_end + 1, lon_start:lon_end + 1]
    u = surface[surface_dict['U10']][lat_start:lat_end + 1, lon_start:lon_end + 1]
    v = surface[surface_dict['V10']][lat_start:lat_end + 1, lon_start:lon_end + 1]
    z_850 = upper[upper_dict['z'],2,:,:][lat_start:lat_end + 1, lon_start:lon_end + 1]/9.80665
    z_200 = upper[upper_dict['z'],9,:,:][lat_start:lat_end + 1, lon_start:lon_end + 1]/9.80665
    z_diff = z_200-z_850
    wind_speed = np.sqrt(u**2 + v**2)
    ax.set_title(f'{time_str} (+{predict_interval}h)  predict: {predict_str}', fontsize=20)

    min_lon, min_lat, min_position = plot_min_value(ax, mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid,
                                                    wind_speed, predict_str, init_str, init_pos, min_position)
    setup_map(ax)
    weather_map_contour(ax, lon_grid, lat_grid, mslp)

    # contourf = ax.contourf(lon_grid, lat_grid, vorticity, cmap='seismic', levels = np.linspace(-10**-4,10**-4,21),extend='both')
    # contourf = ax.contourf(lon_grid, lat_grid, wind_speed, cmap='jet', levels = np.linspace(0,30,7), alpha=0.5)
    contourf = ax.contourf(lon_grid, lat_grid, z_diff, cmap='jet', levels = np.linspace(10000,11500,16), alpha=0.5)
    cbar = plt.colorbar(contourf, ax=ax, orientation='horizontal', extend='both', fraction=0.046, pad=0.04)
    # cbar.set_label('Vorticity', fontsize=16)  # You can customize the label here
    cbar.set_label('200-850hPa Thickness(m)', fontsize=16)  # You can customize the label here
    # cbar.set_label('10m Wind Speed(m/s)', fontsize=16)  # You can customize the label here
    
    plt.show()



#%% multi time
predict_interval_list = np.arange(0,24*7+1,6)[1:]  #볼 예측 시간 지정
year = ['2022']
month = ['08']
day = ['26','27']
times = ['00','06','12','18']
pred_str = '2022.09.01 00UTC'
area = [90, 0, -90, 360]
time_str_input = f'{year[0]}.{month[0]}.{day[0]} {times[0]}UTC_{year[-1]}.{month[-1]}.{day[-1]} {times[-1]}UTC'
time_len = len(year)*len(month)*len(day)*len(times)


#동반구는 0~180, 서반구는 180~360y
lat_start, lat_end, lon_start, lon_end, extent, latlon_ratio = latlon_extent(110,160,5,45)  

pres=500                                            #살펴볼 기압면 결정
p=pres_list.index(str(pres))


fig, axs = plt.subplots(1, 2, figsize=(10*latlon_ratio*4, 10*2), subplot_kw={'projection': proj})
for predict_interval in predict_interval_list:


    # 위경도 범위 설정
    lat_start, lat_end, lon_start, lon_end, extent, latlon_ratio = latlon_extent(110, 160, 5, 45)
    lon_grid, lat_grid = np.meshgrid(lon_indices[lon_start:lon_end + 1], lat_indices[lat_start:lat_end + 1])

    # 최저값 위치를 저장할 리스트
    # min_position = []

    # input_data_dir = rf'{pangu_dir}/input_data/{time_str_input}'
    # surface = np.load(os.path.join(input_data_dir, 'surface.npy')).astype(np.float32)[time_step::time_len]  
    # upper = np.load(os.path.join(input_data_dir, 'upper.npy')).astype(np.float32)[time_step::time_len]  


    # fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
    # mslp = surface[surface_dict['MSLP']][lat_start:lat_end + 1, lon_start:lon_end + 1]
    # ax.set_title(f'{time_str} (+0h)', fontsize=20)

    # min_lon, min_lat = plot_min_value(ax, mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, min_position)
    # setup_map(ax)
    # weather_map_contour(ax, lon_grid, lat_grid, mslp)


    # u = upper[upper_dict['u'],2,:,:][lat_start:lat_end + 1, lon_start:lon_end + 1]
    # v = upper[upper_dict['v'],2,:,:][lat_start:lat_end + 1, lon_start:lon_end + 1]
    # z_850 = upper[upper_dict['z'],2,:,:][lat_start:lat_end + 1, lon_start:lon_end + 1]/9.80665
    # z_200 = upper[upper_dict['z'],9,:,:][lat_start:lat_end + 1, lon_start:lon_end + 1]/9.80665
    # z_diff = z_200-z_850

    # vorticity = calculate_vorticity(u, v, lat_grid, lon_grid)

    # # contourf = ax.contourf(lon_grid, lat_grid, vorticity, cmap='seismic', levels = np.linspace(-10**-4,10**-4,21),extend='both')
    # contourf = ax.contourf(lon_grid, lat_grid, z_diff, cmap='jet')
    # # Add a colorbar
    # cbar = plt.colorbar(contourf, ax=ax, orientation='vertical', extend='both')
    # cbar.set_label('Vorticity', fontsize=16)  # You can customize the label here
    # cbar.set_label('200-850hPa Thickness(m)', fontsize=16)  # You can customize the label here

    # plt.show()

    min_position = []

    for y, m, d, tm in itertools.product(year, month, day, times):
        # time_str = generate_time_str(time_step)
        # print(time_str)
        time_str = time_str = f'{y}/{m}/{d}/{tm}UTC'
        time_str = time_str = f'{y}.{m}.{d} {tm}UTC'
        ax = axs.flatten()[time_step]
        
        time_obj = datetime.strptime(time_str, "%Y.%m.%d %HUTC")
        predict_time = time_obj + timedelta(hours=int(predict_interval))
        predict_str = predict_time.strftime("%Y.%m.%d %HUTC")
        # print(predict_str)

        if predict_str == pred_str:
            # fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
            output_data_dir = rf'{pangu_dir}/output_data/{time_str}'
            
            # Load surface data
            surface = np.load(os.path.join(output_data_dir, rf'surface/{predict_interval}h.npy')).astype(np.float32)
            upper = np.load(os.path.join(output_data_dir, rf'upper/{predict_interval}h.npy')).astype(np.float32) 
            mslp = surface[surface_dict['MSLP']][lat_start:lat_end + 1, lon_start:lon_end + 1]
            u = upper[upper_dict['u'],2,:,:][lat_start:lat_end + 1, lon_start:lon_end + 1]
            v = upper[upper_dict['v'],2,:,:][lat_start:lat_end + 1, lon_start:lon_end + 1]
            vorticity = calculate_vorticity(u, v, lat_grid, lon_grid)

            z_850 = upper[upper_dict['z'],2,:,:][lat_start:lat_end + 1, lon_start:lon_end + 1]/9.80665
            z_200 = upper[upper_dict['z'],9,:,:][lat_start:lat_end + 1, lon_start:lon_end + 1]/9.80665
            z_diff = z_200-z_850

            ax.set_title(f'{time_str} (+{predict_interval}h)/npredict: {predict_str}', fontsize=20)

            min_lon, min_lat = plot_min_value(ax, mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, min_position)
            setup_map(ax)
            weather_map_contour(ax, lon_grid, lat_grid, mslp)

            # contourf = ax.contourf(lon_grid, lat_grid, vorticity, cmap='seismic', levels = np.linspace(-10**-4,10**-4,21),extend='both')
            contourf = ax.contourf(lon_grid, lat_grid, z_diff, cmap='jet')
            # Add a colorbar
            cbar = plt.colorbar(contourf, ax=ax, orientation='vertical', extend='both')
            # cbar.set_label('Vorticity', fontsize=16)  # You can customize the label here
            cbar.set_label('200-850hPa Thickness(m)', fontsize=16)  # You can customize the label here


            # 저장된 최저값 위치에 마커 추가 및 선으로 연결
            for i,(lon, lat) in enumerate(min_position):
                if i==(np.size(min_position)/2-1):
                    ax.scatter(lon, lat, color='blue', transform=proj, marker='x', s=40)
                    ax.scatter(lon, lat, color='blue', transform=proj, facecolors='none', marker = 'o', s=40)
                    
                else:
                    ax.scatter(lon, lat, color='blue', transform=proj)

            # 선으로 연결
            if len(min_position) > 1:
                lons, lats = zip(*min_position)
                ax.plot(lons, lats, color='red', transform=proj, linestyle='-', marker='')
            
            # plt.show()
    
    
    
    

#%%
min_position

