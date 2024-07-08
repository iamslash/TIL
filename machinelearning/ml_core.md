- [Abstract](#abstract)
- [Chapter 1: Real-valued Circuits](#chapter-1-real-valued-circuits)
  - [핵심 개념](#핵심-개념)
  - [자바스크립트 예제](#자바스크립트-예제)
    - [1단계: 유닛 구조 정의](#1단계-유닛-구조-정의)
    - [2단계: 게이트 클래스 정의](#2단계-게이트-클래스-정의)
    - [3단계: 신경망의 순전파 및 역전파 구현](#3단계-신경망의-순전파-및-역전파-구현)
    - [4단계: 역전파 수행](#4단계-역전파-수행)
    - [5단계: 입력값 조정 및 새로운 출력 확인](#5단계-입력값-조정-및-새로운-출력-확인)
- [Chapter 2: Machine Learning](#chapter-2-machine-learning)
  - [핵심 개념](#핵심-개념-1)
  - [자바스크립트 예제](#자바스크립트-예제-1)
    - [1단계: 회로 정의](#1단계-회로-정의)
    - [2단계: SVM 클래스 정의](#2단계-svm-클래스-정의)
    - [3단계: 데이터 준비 및 학습](#3단계-데이터-준비-및-학습)
  - [요약](#요약)

-----

# Abstract

Andrej Karpathy 가 작성한 [Hacker's guide to Neural Networks](https://karpathy.github.io/neuralnets/) 를 읽고 정리한다.

# Chapter 1: Real-valued Circuits

## 핵심 개념

- 실수 회로: 신경망을 실수 값을 가진 회로로 생각합니다. 값들이 곱셈, 덧셈, 최대값 같은 게이트를 통해 흐릅니다.
- 순전파(Forward Pass): 입력값을 게이트를 통해 전달하여 출력값을 계산합니다.
- 기울기 계산: 출력값을 높이기 위해 입력값을 어떻게 조정해야 하는지를 계산합니다. 세 가지 전략이 있습니다:
  - 랜덤 로컬 서치(Random Local Search)
  - 수치적 기울기(Numerical Gradient)
  - 분석적 기울기(Analytic Gradient)
- 역전파(Backpropagation): 체인 룰(Chain Rule)을 사용하여 효율적으로 기울기를 계산하고 네트워크를 통해 기울기를 전달합니다.

## 자바스크립트 예제

### 1단계: 유닛 구조 정의

```js
// 값과 기울기를 저장하는 유닛 구조 정의
class Unit {
  constructor(value, grad) {
    this.value = value; // 순전파 시 계산된 값
    this.grad = grad;   // 역전파 시 계산된 기울기
  }
}
```


### 2단계: 게이트 클래스 정의

```js
// 곱셈 게이트
class MultiplyGate {
  forward(u0, u1) {
    this.u0 = u0;
    this.u1 = u1;
    this.utop = new Unit(u0.value * u1.value, 0.0);
    return this.utop;
  }
  backward() {
    this.u0.grad += this.u1.value * this.utop.grad;
    this.u1.grad += this.u0.value * this.utop.grad;
  }
}

// 덧셈 게이트
class AddGate {
  forward(u0, u1) {
    this.u0 = u0;
    this.u1 = u1;
    this.utop = new Unit(u0.value + u1.value, 0.0);
    return this.utop;
  }
  backward() {
    this.u0.grad += 1 * this.utop.grad;
    this.u1.grad += 1 * this.utop.grad;
  }
}

// 시그모이드 게이트
class SigmoidGate {
  sig(x) {
    return 1 / (1 + Math.exp(-x));
  }
  forward(u0) {
    this.u0 = u0;
    this.utop = new Unit(this.sig(this.u0.value), 0.0);
    return this.utop;
  }
  backward() {
    const s = this.sig(this.u0.value);
    this.u0.grad += (s * (1 - s)) * this.utop.grad;
  }
}
```

### 3단계: 신경망의 순전파 및 역전파 구현

```js
// 입력 유닛 생성
var a = new Unit(1.0, 0.0);
var b = new Unit(2.0, 0.0);
var c = new Unit(-3.0, 0.0);
var x = new Unit(-1.0, 0.0);
var y = new Unit(3.0, 0.0);

// 게이트 생성
var mulg0 = new MultiplyGate();
var mulg1 = new MultiplyGate();
var addg0 = new AddGate();
var addg1 = new AddGate();
var sg0 = new SigmoidGate();

// 순전파 수행
var forwardNeuron = function() {
  var ax = mulg0.forward(a, x); // a*x = -1
  var by = mulg1.forward(b, y); // b*y = 6
  var axpby = addg0.forward(ax, by); // a*x + b*y = 5
  var axpbypc = addg1.forward(axpby, c); // a*x + b*y + c = 2
  var s = sg0.forward(axpbypc); // sig(a*x + b*y + c) = 0.8808
  return s;
};

var s = forwardNeuron();
console.log('회로 출력: ' + s.value); // 출력값 0.8808
```

### 4단계: 역전파 수행

```js
// 역전파를 통해 기울기 계산
s.grad = 1.0;
sg0.backward(); // axpbypc에 대한 기울기 계산
addg1.backward(); // axpby 및 c에 대한 기울기 계산
addg0.backward(); // ax 및 by에 대한 기울기 계산
mulg1.backward(); // b 및 y에 대한 기울기 계산
mulg0.backward(); // a 및 x에 대한 기울기 계산
```

### 5단계: 입력값 조정 및 새로운 출력 확인

```js
var step_size = 0.01;
a.value += step_size * a.grad;
b.value += step_size * b.grad;
c.value += step_size * c.grad;
x.value += step_size * x.grad;
y.value += step_size * y.grad;

s = forwardNeuron();
console.log('역전파 후 회로 출력: ' + s.value); // 새로운 출력값 0.8825
위 코드는 역전파를 통해 기울기를 계산하고 입력값을 조정하여 출력값을 높이는 과정을 보여줍니다. 각 단계는 주석을 통해 설명되어 있으며, 기본적인 신경망 학습 원리를 이해하는 데 도움이 됩니다.
```

# Chapter 2: Machine Learning

## 핵심 개념

- 이진 분류(Binary Classification): 가장 기본적이면서도 실용적인 문제로, +1 또는 -1로 라벨링된 데이터셋을 분류하는 것입니다.
- 훈련 프로토콜(Training Protocol): 데이터셋에서 무작위로 선택된 데이터 포인트를 통해 회로를 학습시키는 과정입니다.
- 확률적 경사 하강법(Stochastic Gradient Descent): 각 데이터 포인트에 대해 회로의 출력을 조정하여 점진적으로 회로의 매개변수를 학습시키는 방법입니다.
- 서포트 벡터 머신(SVM): 선형 분류기 중 하나로, 이 장에서는 힘의 사양(force specification) 관점에서 설명됩니다.

## 자바스크립트 예제

### 1단계: 회로 정의

먼저 회로를 정의합니다. 이 회로는 입력 x, y와 매개변수 a, b, c를 받아 ax + by + c를 계산하고, 기울기도 계산할 수 있습니다.

```js
// 회로 클래스 정의
var Circuit = function() {
  this.mulg0 = new MultiplyGate();
  this.mulg1 = new MultiplyGate();
  this.addg0 = new AddGate();
  this.addg1 = new AddGate();
};

Circuit.prototype = {
  forward: function(x, y, a, b, c) {
    this.ax = this.mulg0.forward(a, x); // a * x 계산
    this.by = this.mulg1.forward(b, y); // b * y 계산
    this.axpby = this.addg0.forward(this.ax, this.by); // a * x + b * y 계산
    this.axpbypc = this.addg1.forward(this.axpby, c); // a * x + b * y + c 계산
    return this.axpbypc;
  },
  backward: function(gradient_top) {
    this.axpbypc.grad = gradient_top;
    this.addg1.backward();
    this.addg0.backward();
    this.mulg1.backward();
    this.mulg0.backward();
  }
};
```

### 2단계: SVM 클래스 정의

다음으로 SVM 클래스를 정의합니다. 이 클래스는 랜덤 초기 매개변수를 사용하고, 회로를 이용하여 데이터를 학습시킵니다.

```js
// SVM 클래스 정의
var SVM = function() {
  this.a = new Unit(1.0, 0.0); 
  this.b = new Unit(-2.0, 0.0);
  this.c = new Unit(-1.0, 0.0);
  this.circuit = new Circuit();
};

SVM.prototype = {
  forward: function(x, y) {
    this.unit_out = this.circuit.forward(x, y, this.a, this.b, this.c);
    return this.unit_out;
  },
  backward: function(label) {
    this.a.grad = 0.0; 
    this.b.grad = 0.0; 
    this.c.grad = 0.0;
    var pull = 0.0;
    if (label === 1 && this.unit_out.value < 1) pull = 1;
    if (label === -1 && this.unit_out.value > -1) pull = -1;
    this.circuit.backward(pull);
    this.a.grad += -this.a.value;
    this.b.grad += -this.b.value;
  },
  learnFrom: function(x, y, label) {
    this.forward(x, y);
    this.backward(label);
    this.parameterUpdate();
  },
  parameterUpdate: function() {
    var step_size = 0.01;
    this.a.value += step_size * this.a.grad;
    this.b.value += step_size * this.b.grad;
    this.c.value += step_size * this.c.grad;
  }
};
```

### 3단계: 데이터 준비 및 학습

이제 데이터셋을 준비하고 SVM을 학습시킵니다.

```js
// 데이터셋 및 라벨 준비
var data = []; var labels = [];
data.push([1.2, 0.7]); labels.push(1);
data.push([-0.3, -0.5]); labels.push(-1);
data.push([3.0, 0.1]); labels.push(1);
data.push([-0.1, -1.0]); labels.push(-1);
data.push([-1.0, 1.1]); labels.push(-1);
data.push([2.1, -3]); labels.push(1);

var svm = new SVM();

// 학습 정확도 평가 함수
var evalTrainingAccuracy = function() {
  var num_correct = 0;
  for (var i = 0; i < data.length; i++) {
    var x = new Unit(data[i][0], 0.0);
    var y = new Unit(data[i][1], 0.0);
    var true_label = labels[i];
    var predicted_label = svm.forward(x, y).value > 0 ? 1 : -1;
    if (predicted_label === true_label) {
      num_correct++;
    }
  }
  return num_correct / data.length;
};

// 학습 루프
for (var iter = 0; iter < 400; iter++) {
  var i = Math.floor(Math.random() * data.length);
  var x = new Unit(data[i][0], 0.0);
  var y = new Unit(data[i][1], 0.0);
  var label = labels[i];
  svm.learnFrom(x, y, label);
  if (iter % 25 === 0) {
    console.log('Iteration ' + iter + ': Training accuracy: ' + evalTrainingAccuracy());
  }
}
```

## 요약

- 이진 분류 문제: +1 또는 -1로 라벨링된 데이터 포인트를 분류합니다.
- 훈련 프로토콜: 확률적 경사 하강법을 사용하여 매개변수를 조정합니다.
- SVM: 선형 분류기로, 매개변수 a, b, c를 학습시켜 데이터 포인트를 분류합니다.
- 역전파: 기울기를 계산하여 매개변수를 업데이트합니다.

이 과정을 통해 간단한 SVM을 학습시키는 방법을 이해할 수 있습니다. 이 예제를 통해 회로와 기울기 계산이 머신러닝에서 어떻게 사용되는지 배웠습니다.
