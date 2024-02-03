# Pangu Weather
폴더 구조는 다음과 같은 하위계층을 따른다.
```
├── root
│   ├── input_data
│   │   ├── YYYY.MM.DD HHUTC(multi_time의 경우 YYYY-MM-DD-HHUTC로 구성)
│   │   │   ├── surface.npy
│   │   │   ├── upper.npy
│   ├── output_data
│   │   ├── YYYY.MM.DD HHUTC(multi_time의 경우 YYYY-MM-DD-HHUTC로 구성)
│   │   │   ├── surface
│   │   │   │   ├── HHHh.npy
│   │   │   ├── upper
│   │   │   │   ├── HHHh.npy
│   ├── pangu_weather_1.onnx
│   ├── pangu_weather_3.onnx
│   ├── pangu_weather_6.onnx
│   ├── pangu_weather_24.onnx
│   ├── inference_cpu.py
│   ├── inference_gpu.py
│   ├── inference_iterative.py
```

cpu 이용시 cmd나 prompt에 다음을 입력(requirements_cpu.txt의 위치에서 실행해야 함)
```
pip install -r requirements_cpu.txt
```

GPU 이용시 CUDA 11.6 버전, Linux이면 8.2.4 버전, Window이면 cudnn 8.5.0.96 버전을 설치하고 다음을 실행
<br/>(CUDA가 굉장히 크므로 아직 다루지 않았음)
```
pip install -r requirements_gpu.txt
```

필요한 패키지 다운
```
pip install ai-models-panguweather
```

[Pangu Git](https://github.com/198808xc/Pangu-Weather?tab=readme-ov-file#downloading-trained-models)에 나온 onnx 파일들도 다운 받는다.

## ERA5_download.py
원하는 ERA5 시간 데이터를 Pangu input 형식에 맞게 변형하는 코드 
<br/> ERA5 개인 url, key 필요
<br/> 각 변수들은 [Pangu Git](https://github.com/198808xc/Pangu-Weather?tab=readme-ov-file#downloading-trained-models)의 설명을 참조하라.
<br/> 시간대 조정은 다음과 같이 한다.
```python
time_str = 'YYYY.MM.DD HHUTC'
```

## inference_cpu.py, inference_iterative.py
time_str, predict_interval을 조정하여 초기장 시각과 예측 시간(1,3,6,24 중) 조정
<br/>inference_iterative.py에서는 예측 시간을 조합하여 더 먼 미래의 예측장 도출

## plotting.py
time_str, predict_interval_list을 조정하여 초기장 시각과 예측 시간 리스트 조정  
latlon_extent(0,360,-90,90)에서 살펴볼 위경도 범위 조정  
pres로 기압면 결정정  
***wind vector map 범례 표시 위치가 이상하게 표기됨 어떻게 해결해야할지??***




      
이미지 예시
![image](https://github.com/jjoo0727/Convective-Systems-Tropical-Dynamics-Laboratory/assets/63052158/9dbac0d2-23b2-4d7e-9871-b21830442bb4)
![image](https://github.com/jjoo0727/Convective-Systems-Tropical-Dynamics-Laboratory/assets/63052158/f3971dc9-caa8-4c04-a992-e6dac09571b3)


