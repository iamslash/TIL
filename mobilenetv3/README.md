
# Basic

MobileNetV3는 이미지 분류, 객체 탐지 등의 작업에 사용되는 경량화된 딥러닝 모델입니다. 이 모델은 특히 모바일이나 임베디드 시스템과 같이 연산 능력이 제한된 환경에서 효율적으로 동작하도록 설계되었습니다. MobileNetV3는 MobileNet 시리즈 중 하나로, 이전 버전인 MobileNetV1과 MobileNetV2의 아이디어를 발전시켜 더욱 향상된 성능과 효율성을 제공합니다.

핵심 개념

- 경량화된 컨볼루션 블록: MobileNetV3는 작은 크기의 파라미터와 연산량으로도 높은 성능을 내기 위해 Depthwise Separable Convolution을 사용합니다. 이는 표준 컨볼루션 연산을 깊이 방향의 컨볼루션과 포인트 방향의 컨볼루션으로 분리하여 연산량을 크게 줄입니다.
- 하드 스위시(Hard-Swish) 활성화 함수: 새로운 활성화 함수인 하드 스위시는 모델의 비선형성을 증가시키면서도 계산 비용은 낮춰 모델의 효율성을 개선합니다.
- 네트워크 구조 최적화: MobileNetV3는 머신러닝 기법을 사용하여 모델 구조의 각 부분을 최적화합니다. 예를 들어, 네트워크의 입력 및 출력 채널 수, 활성화 함수의 사용 여부 등을 데이터 기반으로 최적화합니다.

PyTorch에서 MobileNetV3 모델을 사용하는 예제 코드를 보여드리겠습니다. 이 코드는 PyTorch의 torchvision.models 모듈을 사용하여 MobileNetV3 모델을 로드하고, 간단한 이미지 분류 작업을 수행합니다.

```py
import torch
from torchvision import models, transforms
from PIL import Image

# MobileNetV3 모델 로드 (사전 학습된 가중치 사용)
model = models.mobilenet_v3_large(pretrained=True)
model.eval()  # 평가 모드로 설정

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 이미지 로드 및 전처리
img = Image.open("path/to/your/image.jpg")  # 이미지 경로
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# 예측
with torch.no_grad():
    output = model(batch_t)

# 결과 해석
_, predicted = torch.max(output, 1)
print(f'Predicted: {predicted.item()}')  # 예측된 클래스 인덱스 출력
```

이 코드는 사전 학습된 MobileNetV3 모델을 로드하여, 주어진 이미지에 대한 분류 예측을 수행합니다. 이미지 전처리 단계에서는 입력 이미지를 모델이 요구하는 형식으로 변환합니다. 예측 단계에서는 모델을 통해 이미지를 전달하고, 가장 높은 예측 값을 가진 클래스를 출력합니다.
