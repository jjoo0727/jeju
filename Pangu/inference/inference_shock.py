
#%%
import torch
import onnxruntime as ort
import os
import numpy as np
import onnx
import time
import itertools

lat_indices = np.linspace(90, -90, 721)
lon_indices = np.linspace(-180, 180, 1441)[:-1]

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

lat_start, lat_end, lon_start, lon_end, extent, latlon_ratio = latlon_extent(100,160,5,45)  


year = ['2022']
month = ['08']
day = ['27']
times = ['00']
shock_list = [10,100,1000]
level_list = range(13)
surface_factor_list = []    # 예시: 지표면에서는 'MSLP'만 선택
upper_factor_list = [['z'],['t'],['q']] 
# surface_factors.sort()
# upper_factors.sort()
# surface_str = "".join([f"_{factor}" for factor in surface_factors])  # 각 요소 앞에 _ 추가
# upper_str = "".join([f"_{factor}" for factor in upper_factors])  # 각 요소 앞에 _ 추가
pangu_dir = r'/Data/home/jjoo0727/Pangu-Weather'

surface_factor = ['MSLP', 'U10', 'V10', 'T2M']
surface_dict = {'MSLP':0, 'U10':1, 'V10':2, 'T2M':3}
upper_factor = ['z', 'q', 't', 'u', 'v']
upper_dict = {'z':0, 'q':1, 't':2, 'u':3, 'v':4}


# Set the behavior of onnxruntime
options = ort.SessionOptions()
options.enable_cpu_mem_arena= True
options.enable_mem_pattern = False
options.enable_mem_reuse = False

# Increase the number for faster inference and more memory consumption
# options.intra_op_num_threads = 1

# Set the behavior of cuda provider for the first GPU
cuda_provider_options_gpu0 = {'arena_extend_strategy': 'kSameAsRequested', 'device_id': 0}

# Set the behavior of cuda provider for the second GPU
cuda_provider_options_gpu1 = {'arena_extend_strategy': 'kSameAsRequested', 'device_id': 1}

# Initialize onnxruntime session for Pangu-Weather Models on different GPUs
ort_session_6 = ort.InferenceSession(rf'{pangu_dir}/pangu_weather_6.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options_gpu0)])
ort_session_24 = ort.InferenceSession(rf'{pangu_dir}/pangu_weather_24.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options_gpu1)])



start = time.time()

for upper_factor in upper_factor_list:
    for shock in shock_list:
        for level in level_list:
            for y, m, d, tm in itertools.product(year, month, day, times):
                time_str = f'{y}/{m}/{d}/{tm}UTC'

                input_data_dir = rf'{pangu_dir}/input_data/{time_str}'
                output_data_dir = rf'{pangu_dir}/output_data/{time_str}'

                input_upper = np.load(os.path.join(input_data_dir, 'upper.npy')).astype(np.float32)
                input_surface = np.load(os.path.join(input_data_dir, 'surface.npy')).astype(np.float32)


                

                upper_str = "".join([f"_{factor}" for factor in upper_factor])


                output_data_dir = rf'{pangu_dir}/output_data/{time_str}/{upper_str}SH_{shock}/{level}'
                    
                if not os.path.exists(os.path.join(output_data_dir, f'upper')):
                    os.makedirs(os.path.join(output_data_dir, f'upper'))
                if not os.path.exists(os.path.join(output_data_dir, f'surface')):
                    os.makedirs(os.path.join(output_data_dir, f'surface'))
                
                
                perturbed_upper = input_upper.copy()
                perturbed_surface = input_surface.copy()
                # Perturbation 생성 및 적용     
                    

                for factor in upper_factor:
                    idx = upper_dict[factor]
                    # for j in range(13):
                    perturbed_upper[idx,level,int((lat_start+lat_end)/2), int((lon_start+lon_end)/2)] += shock 
                        
                    

                    
                np.save(os.path.join(output_data_dir, f'upper/0h'), perturbed_upper[:,:,lat_start: lat_end+1, lon_start:lon_end+1])
                np.save(os.path.join(output_data_dir, f'surface/0h'), perturbed_surface[:,lat_start: lat_end+1, lon_start:lon_end+1])

                perturbed_24, perturbed_surface_24 = perturbed_upper, perturbed_surface

                for i in range(28):
                    start_i = time.time()
                    predict_interval = 6*(i+1)
                    if (i+1) % 4 == 0:
                        output, output_surface = ort_session_24.run(None, {'input':perturbed_24, 'input_surface':perturbed_surface_24})
                        perturbed_24, perturbed_surface_24 = output, output_surface
                        np.save(os.path.join(output_data_dir, f'upper/{predict_interval}h'), output[:,:,lat_start: lat_end+1, lon_start:lon_end+1])
                        np.save(os.path.join(output_data_dir, f'surface/{predict_interval}h'), output_surface[:,lat_start: lat_end+1, lon_start:lon_end+1])
                        


                    # 6시간 간격도 저장하고 싶으면 주석 해제
                    else:
                        output, output_surface = ort_session_6.run(None, {'input':perturbed_upper, 'input_surface':perturbed_surface})
                        np.save(os.path.join(output_data_dir, f'upper/{predict_interval}h'), output[:,:,lat_start: lat_end+1, lon_start:lon_end+1])
                        np.save(os.path.join(output_data_dir, f'surface/{predict_interval}h'), output_surface[:,lat_start: lat_end+1, lon_start:lon_end+1])

                    perturbed_upper, perturbed_surface = output, output_surface
                    end_i = time.time()
                    print(f'{upper_factor} {i+1}번째 반복 +{predict_interval}h {end_i-start_i}s')
                

                end = time.time()
                print(f"{upper_factor}: {end-start}s")