# Pangu Weather
폴더 구조는 다음과 같은 하위계층을 따라야한다.
```
├── root
│   ├── input_data
│   │   ├── YYYY.MM.DD HHUTC
│   │   │   ├── surface.npy
│   │   │   ├── upper.npy
│   ├── output_data
│   │   ├── YYYY.MM.DD HHUTC
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
│   ├── ERA5_download.py
```

cpu 이용시 cmd나 prompt에 다음을 입력(requirements_cpu.txt의 위치에서 실행해야 함)
```
pip install -r requirements_cpu.txt
```

GPU 이용시 CUDA 11.6 버전 및 cudnn 8.5.0.96 버전을 설치하고 다음을 실행
CUDA가 굉장히 크므로 아직 다루지 않았음
```
pip install -r requirements_gpu.txt
```

https://github.com/198808xc/Pangu-Weather?tab=readme-ov-file#downloading-trained-models에 나온 onnx 파일들도 다운 받는다.

