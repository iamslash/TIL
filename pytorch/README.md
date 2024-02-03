- [Materials](#materials)
- [Image Classification Implementation](#image-classification-implementation)
  - [Simple Image Classification Training](#simple-image-classification-training)
  - [Simple Image Classification Training With Checkpoints](#simple-image-classification-training-with-checkpoints)
  - [Simple Image Classification Exporting for Triton Serving](#simple-image-classification-exporting-for-triton-serving)
  - [Simple Image Classification Evaluation](#simple-image-classification-evaluation)
- [Torchscript](#torchscript)

-----

# Materials

- [PyTorch 강좌](https://076923.github.io/posts/Python-pytorch-1/)

# Image Classification Implementation

## Simple Image Classification Training

```py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 데이터셋 로드 및 전처리
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# CNN 모델 정의
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# 손실 함수 및 최적화기 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 모델 훈련
for epoch in range(2):  # 데이터셋을 여러 번 반복

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

**optimizer.zero_grad() 에서 하는일**

`optimizer.zero_grad()` 호출의 목적을 이해하기 위해, 먼저 딥러닝에서 모델을 훈련시키는 과정에 대해 간단히 알아보겠습니다. 딥러닝 모델을 훈련시키는 과정은 기본적으로 모델이 데이터로부터 배우도록 만드는 일련의 반복적인 단계들로 구성됩니다. 이 과정에서 모델은 데이터를 보고, 예측을 하고, 예측이 얼마나 잘못되었는지를 평가하며, 그 정보를 사용해 자신을 조금씩 개선해 나갑니다.

이제 optimizer.zero_grad()가 필요한 이유를 살펴보겠습니다:

기울기(Gradient)란?

모델을 훈련시킬 때, "기울기"는 모델의 예측이 실제 값과 얼마나 다른지, 그리고 그 오차를 최소화하기 위해 모델의 가중치를 어떻게 조정해야 하는지를 알려주는 방향과 크기를 가진 벡터입니다. 각 훈련 단계에서 모델의 가중치에 대한 오차의 기울기를 계산하고, 이 기울기를 사용해 모델의 가중치를 조금씩 조정합니다.

기울기 축적(Accumulation)

PyTorch와 같은 대부분의 딥러닝 프레임워크에서는 기울기를 자동으로 계산해주는 자동 미분 기능을 제공합니다. 특히, PyTorch는 기본적으로 기울기를 "축적"하는 방식으로 동작합니다. 즉, 각 배치(batch)를 처리할 때마다 계산된 기울기를 이전 기울기에 더해 나가는 방식입니다. 이는 일부 고급 기능을 구현할 때 유용할 수 있지만, 대부분의 일반적인 훈련 상황에서는 원치 않는 동작일 수 있습니다.

`optimizer.zero_grad()`의 역할

optimizer.zero_grad()를 호출하는 것은 훈련의 각 단계가 시작될 때 마다 이전 단계에서 계산된 기울기를 "제거"하고 새로운 출발점에서 시작하겠다는 것을 의미합니다. 이는 모델이 각 훈련 단계에서 오직 현재 데이터 배치만을 바탕으로 가중치를 조정하도록 보장합니다. 만약 이를 호출하지 않는다면, 기울기가 계속 축적되어 모델의 학습 과정이 의도하지 않은 방식으로 이루어질 수 있으며, 결국 잘못된 방향으로 가중치가 업데이트될 가능성이 있습니다.

"기울기를 0으로 만드는 것은 새로운 게임을 시작하기 전에 점수판을 초기화하는 것과 같아. 만약 점수를 초기화하지 않는다면, 이전 게임의 점수가 계속 누적되어, 각 게임의 실제 점수를 제대로 파악하기 어려워져. 마찬가지로, 모델을 훈련시킬 때 마다 optimizer.zero_grad()를 호출하는 것은 모델이 '깨끗한 상태'에서 시작해, 오직 현재 데이터로부터만 배우도록 하는 거야."

이렇게 기울기를 0으로 설정하는 단계는 모델을 올바르게 훈련시키기 위한 중요한 과정 중 하나입니다.

**loss.backward() 에서 하는 일**

loss.backward() 호출은 PyTorch에서 매우 중요한 작업을 수행합니다. 이 함수는 네트워크의 모든 학습 가능한 파라미터(가중치)에 대해 손실 함수의 기울기를 자동으로 계산하는 역전파(backpropagation) 과정을 시작합니다. 이를 통해 네트워크를 훈련시키기 위한 기울기(gradient)가 계산됩니다. 이 과정을 조금 더 쉽게 이해하기 위해, 몇 가지 주요 개념을 간단히 설명하겠습니다.

역전파(Backpropagation)란?

역전파는 신경망을 훈련시키기 위해 손실 함수에서 입력층으로 가중치를 조정하는 방향과 크기를 결정하기 위한 기울기를 계산하는 방법입니다. 기본적으로, 네트워크의 출력과 실제 값 사이의 오차(손실)를 줄이기 위해, 이 오차를 네트워크의 각 층을 거슬러 올라가며 분배합니다. 이 과정에서 각 파라미터(가중치)에 대한 손실 함수의 기울기가 계산됩니다.

loss.backward()의 역할

- 기울기 계산: loss.backward()는 손실 함수의 결과(스칼라 값)를 바탕으로, 네트워크 파라미터에 대한 손실 함수의 기울기를 계산합니다. 이 기울기는 파라미터를 어떻게 조정해야 손실을 줄일 수 있는지를 나타냅니다.
- 역전파 실행: 이 함수는 계산된 기울기를 사용하여 네트워크를 거슬러 올라가며 각 파라미터에 대한 기울기를 저장합니다. 이 과정은 네트워크의 출력층에서부터 시작하여 입력층으로 진행됩니다.
- 가중치 업데이트 준비: 기울기가 계산되고 저장된 후, 이를 사용하여 네트워크의 가중치를 업데이트할 수 있습니다. 실제 가중치 업데이트는 optimizer.step() 호출을 통해 수행됩니다. loss.backward()는 가중치를 직접 업데이트하지 않지만, 가중치를 어떻게 조정해야 할지에 대한 방향과 크기를 제공합니다.

작동 원리

loss.backward() 호출 시, PyTorch는 자동 미분(autograd) 시스템을 사용하여 네트워크의 각 학습 가능한 파라미터에 대한 손실 함수의 기울기를 계산합니다. 각 파라미터에 대해, 손실이 파라미터의 작은 변화에 얼마나 민감한지를 나타내는 값이 계산되며, 이는 파라미터를 최적화하는 데 사용됩니다.

이 과정은 모델이 어떤 파라미터를 조정해야 손실을 줄이고, 학습 목표에 더 가까워질 수 있는지를 알아내는 데 필수적입니다. 따라서, 모든 훈련 단계에서 loss.backward()를 호출하여 기울기를 계산하고, 이어서 optimizer.step()을 호출하여 계산된 기울기에 따라 가중치를 조정합니다.

**optimizer.step() 에서 하는 일**

optimizer.step() 함수는 PyTorch에서 모델의 파라미터(가중치)를 실제로 업데이트하는 단계입니다. 이 함수는 loss.backward()에 의해 계산된 기울기를 사용하여 각 파라미터를 업데이트합니다. 이 과정은 모델을 훈련시키는 데 있어서 핵심적인 부분으로, 모델의 성능을 개선하기 위해 파라미터를 조정하는 방법을 정의합니다.

작동 원리

optimizer.step() 호출이 일어나기 전에, 모델의 각 파라미터에 대한 손실 함수의 기울기는 loss.backward() 호출을 통해 계산되어야 합니다. 이 기울기는 모델이 예측을 개선하기 위해 각 파라미터를 얼마나, 그리고 어떤 방향으로 조정해야 하는지를 나타냅니다.

optimizer.step() 함수는 다음과 같은 과정을 수행합니다:

- 가중치 업데이트: 이 함수는 저장된 기울기를 사용하여 각 파라미터(가중치)를 업데이트합니다. 업데이트는 일반적으로 다음의 간단한 수식을 사용하여 수행됩니다: 
    ```
    새 가중치 = 현재 가중치 − 학습률 × 기울기
    ```

여기서 학습률(learning rate)은 각 스텝에서 파라미터를 얼마나 조정할 것인지를 결정하는 하이퍼파라미터입니다. 학습률은 학습 과정의 속도와 효율성에 큰 영향을 미칩니다.

- 최적화 알고리즘 적용: PyTorch는 다양한 최적화 알고리즘을 제공합니다(예: SGD, Adam, RMSprop 등). optimizer.step()는 설정된 최적화 알고리즘에 따라 파라미터 업데이트 방식을 결정합니다. 각 최적화 알고리즘은 기울기와 다른 요소들(예: 이전 기울기의 이동 평균 등)을 사용하여 가중치 업데이트 방식을 조정합니다.

- 학습 과정 조정: 이 단계는 모델이 학습 데이터에 대해 어떻게 반응하는지를 기반으로 모델의 가중치를 조정합니다. 이를 통해 모델은 점차적으로 손실을 줄이고, 주어진 태스크에 대한 예측 성능을 향상시킬 수 있습니다.

중요성

optimizer.step()은 모델을 훈련시키는 과정에서 매우 중요한 단계입니다. 이 단계 없이는 loss.backward()에 의해 계산된 기울기가 실제 모델의 학습에 사용되지 않습니다. 즉, 모델의 가중치가 업데이트되지 않고, 따라서 모델의 성능이 개선되지 않습니다. 따라서, 모든 훈련 에폭(epoch) 또는 반복(iteration) 후에 optimizer.step()을 호출하여 모델이 학습 데이터로부터 배운 내용을 반영하도록 해야 합니다.

## Simple Image Classification Training With Checkpoints

**Saving Checkpoints**

```py
# Function to save checkpoint
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

# Model training
for epoch in range(2):  # Let's assume you're running for two epochs for simplicity
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # Save checkpoint every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            # Save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            })
            running_loss = 0.0

print('Finished Training')
```

**Loading Checkpoints**

```py
# Function to load checkpoint
def load_checkpoint(filename="checkpoint.pth.tar"):
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

# Example usage
start_epoch = load_checkpoint('checkpoint.pth.tar')  # Load checkpoint
print(f"Resuming training from epoch {start_epoch}")
```

**Putting It All Together**

```py
try:
    start_epoch = load_checkpoint('checkpoint.pth.tar')
    print(f"Resuming training from epoch {start_epoch}")
except FileNotFoundError:
    print("No checkpoint found. Starting from scratch.")
    start_epoch = 0
```

## Simple Image Classification Exporting for Triton Serving

**model.pt**

Create a directory structure for your model in the Triton model repository. For a TorchScript model, this typically involves creating a directory named after your model (my_model) and placing the exported `.pt` file inside a versioned subdirectory (e.g., `my_model/1/model.pt`).

```py
# Assuming the training loop is finished and 'net' is your trained model

# Step 2: Convert to TorchScript
# Prepare an example input for the tracing
example_input = torch.randn(1, 3, 32, 32)  # CIFAR10 image size
traced_script_module = torch.jit.trace(net, example_input)

# Step 3: Save the TorchScript Model
traced_script_module.save("model.pt")

print('Model has been converted to TorchScript and saved.')
```

**model.pbtext**

Triton requires a `config.pbtxt` file to understand how to serve your model. The configuration specifies details like the model name, version policy, input/output tensors, and their data types. For a TorchScript model, the configuration might look like this:

```json
name: "my_model"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
  {
    name: "input"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [3, 32, 32]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [10]
  }
]
```

## Simple Image Classification Evaluation

`정밀도(Precision)`와 `재현율(Recall)`을 그래프로 표현하기 위해서는 먼저 이 두 메트릭을 각 클래스에 대해 계산해야 합니다. CIFAR-10 데이터셋은 10개의 클래스를 가지고 있으므로, 각 클래스에 대한 정밀도와 재현율을 계산한 후, 이를 그래프로 시각화할 수 있습니다.

아래는 정밀도와 재현율을 계산하고 시각화하는 과정을 단계별로 설명한 코드입니다:

**1단계: 정밀도와 재현율 계산**

먼저 모든 클래스에 대한 정밀도와 재현율을 계산합니다. 이를 위해 각 클래스별 `True Positive (TP)`, `False Positive (FP)`, 그리고 `False Negative (FN)` 값을 계산해야 합니다.

**2단계: 그래프로 시각화**

계산된 정밀도와 재현율을 바 그래프로 시각화합니다. Matplotlib 라이브러리를 사용하여 이 과정을 수행할 수 있습니다.

아래 코드는 이러한 과정을 구현한 예시입니다:

```py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

# Set the model to evaluation mode
net.eval()

# Initialize lists to store true labels and predictions
true_labels = []
predictions = []

# Disable gradient computation for evaluation
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        
        _, predicted = torch.max(outputs, 1)
        
        true_labels.extend(labels.numpy())
        predictions.extend(predicted.numpy())

# Convert lists to numpy arrays for use with sklearn metrics
true_labels = np.array(true_labels)
predictions = np.array(predictions)

# Calculate precision and recall for each class
num_classes = 10  # CIFAR-10 has 10 classes
precision = dict()
recall = dict()

for i in range(num_classes):
    precision[i], recall[i], _ = precision_recall_curve(true_labels == i, predictions == i)

# Plot precision and recall for each class
for i in range(num_classes):
    plt.figure()
    plt.step(recall[i], precision[i], where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve for class {}'.format(i))
    plt.show()
```

이 코드는 각 클래스별로 정밀도와 재현율을 계산하고, 이를 바 그래프로 나타냅니다. 실제로 정밀도와 재현율을 계산하기 위해서는 각 클래스별로 TP, FP, FN 값을 정확하게 계산해야 하며, 위의 예시에서는 간략화를 위해 TP와 FN을 동일하게 가정하여 계산하였습니다. 실제 응용에서는 각 값에 대한 정확한 계산이 필요합니다.

# Torchscript

TorchScript의 Tracing과 Scripting은 PyTorch 모델을 TorchScript로 변환하는 두 가지 주요 방법입니다. 각 방법은 서로 다른 사용 사례와 제약 사항을 가지고 있으며, 모델의 특성과 필요에 따라 적절한 방법을 선택해야 합니다.

**Tracing**

Tracing은 모델에 sample 입력을 제공하고, 실행되는 연산을 추적하여 computational graph를 생성합니다. 이 과정은 모델의 동적인 특성을 capture하지 못하는 단점이 있습니다. 즉, 입력에 따라 변하는 control flow(예: if 문, for loop 등)를 가진 모델의 경우 정확히 반영되지 않을 수 있습니다. 그러나 모델이 static하거나 입력에 따른 구조 변화가 없는 경우, Tracing은 빠르고 쉽게 모델을 TorchScript로 변환할 수 있는 방법입니다.

Tracing 예제 코드:

```py
import torch

# 모델 정의
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

# 모델 인스턴스화
model = MyModel()

# Sample 입력 생성
example_input = torch.rand(1, 10)

# Tracing을 사용하여 모델을 TorchScript로 변환
traced_model = torch.jit.trace(model, example_input)

# 변환된 모델 저장
traced_model.save("traced_model.pt")
```

**Scripting**

Scripting은 Python 코드를 분석하여 직접 TorchScript 코드로 변환합니다. 이 방법은 모델의 동적인 특성과 복잡한 control flow를 정확히 처리할 수 있습니다. Scripting은 Python 코드를 TorchScript의 정적 그래프 표현으로 변환하기 때문에, Python에서 사용되는 대부분의 기능을 지원합니다. 복잡한 모델이나 동적인 구조를 가진 모델을 TorchScript로 변환해야 하는 경우에 적합합니다.

Scripting 예제 코드:

```py
import torch

class MyDynamicModel(torch.nn.Module):
    def __init__(self):
        super(MyDynamicModel, self).__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        if x.sum() > 0:
            return self.linear(x)
        else:
            return x

# 모델 인스턴스화
model = MyDynamicModel()

# Scripting을 사용하여 모델을 TorchScript로 변환
scripted_model = torch.jit.script(model)

# 변환된 모델 저장
scripted_model.save("scripted_model.pt")
```

이 예제에서 MyDynamicModel은 입력 데이터의 합에 따라 다른 동작을 하는 동적인 특성을 가지고 있습니다. 이러한 동적인 특성은 Tracing으로는 정확히 capture하기 어렵지만, Scripting을 통해 정확히 TorchScript 코드로 변환할 수 있습니다.

각 방법의 선택은 모델의 특성과 요구 사항에 따라 달라집니다. Tracing은 구현이 간단하고 빠르게 모델을 변환할 수 있는 반면, Scripting은 더 복잡한 모델의 동적인 특성을 정확히 처리할 수 있는 장점이 있습니다.
