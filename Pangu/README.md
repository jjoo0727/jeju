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
