- [Abstract](#abstract)
  - [🚀 KServe란?](#-kserve란)
  - [📦 주요 특징](#-주요-특징)
  - [🧱 핵심 구성 요소](#-핵심-구성-요소)
  - [🧪 InferenceService 예시](#-inferenceservice-예시)
  - [🌐 API 호출 예시](#-api-호출-예시)
  - [🧠 KServe가 해결하는 문제](#-kserve가-해결하는-문제)
  - [📚 지원 모델 유형](#-지원-모델-유형)
  - [✅ 실전에서 KServe는 이렇게 사용됩니다](#-실전에서-kserve는-이렇게-사용됩니다)
  - [💡 KServe vs 대체 도구](#-kserve-vs-대체-도구)
  - [✅ 정리 요약](#-정리-요약)

-----

# Abstract

**KServe**는 Kubernetes 위에서 **AI/ML 모델을 서빙(Serving)하기 위한 표준 플랫폼**입니다.
Google, IBM, Seldon, Bloomberg 등이 공동 개발했고, 현재는 **Kubeflow와 독립된 CNCF 프로젝트**로 발전하고 있습니다.

---

## 🚀 KServe란?

> **KServe**는 다양한 머신러닝/딥러닝 프레임워크 모델을
> \*\*자동 배포, 확장, 모니터링 가능한 추론 서버(서빙 레이어)\*\*로 만들기 위한 Kubernetes 네이티브 오픈소스 플랫폼입니다.

---

## 📦 주요 특징

| 기능                                 | 설명                                                         |
| ---------------------------------- | ---------------------------------------------------------- |
| ✅ 다양한 프레임워크 지원                     | PyTorch, TensorFlow, XGBoost, ONNX, Scikit-learn, Triton 등 |
| ✅ 자동 스케일링                          | 요청 없으면 0으로 줄이고, 요청 오면 자동 확장 (Knative 기반)                   |
| ✅ REST & gRPC API 제공               | 표준 인터페이스로 호출 가능                                            |
| ✅ 배치/실시간 추론 지원                     | 실시간 서비스, 또는 배치용 InferenceJob 가능                            |
| ✅ Model Mesh (multi-model serving) | 여러 모델을 하나의 인스턴스에 올릴 수 있음                                   |
| ✅ A/B 테스트 및 Shadow 테스트             | 다양한 실험 전략 적용 가능                                            |
| ✅ Prometheus & Grafana 연동          | 모니터링 및 메트릭 수집 용이                                           |
| ✅ 모델 저장소 연동                        | S3, GCS, PVC 등에서 모델 자동 로딩                                  |

---

## 🧱 핵심 구성 요소

| 구성 요소                | 설명                                                              |
| -------------------- | --------------------------------------------------------------- |
| **InferenceService** | 모델을 정의하는 Kubernetes CRD(Custom Resource Definition)             |
| **Model Server**     | 실제 추론을 수행하는 서버 (e.g. `Triton`, `TorchServe`, `SKLearnServer` 등) |
| **Knative**          | 서버리스 기능, autoscaling, Istio 연동                                  |
| **Istio / Gateway**  | 트래픽 라우팅, 버전 관리, 보안 (option)                                     |

---

## 🧪 InferenceService 예시

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: my-model
spec:
  predictor:
    sklearn:
      storageUri: s3://my-bucket/models/sklearn-model/
```

→ 위 YAML을 `kubectl apply -f` 하면, KServe가 모델을 자동으로 불러와 추론 API 생성

---

## 🌐 API 호출 예시

```bash
curl -X POST http://<host>/v1/models/my-model:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[1.0, 2.0, 3.0, 4.0]]}'
```

---

## 🧠 KServe가 해결하는 문제

| 전통 서빙 방식 문제    | KServe의 해결책                        |
| -------------- | ---------------------------------- |
| 모델 배포 자동화 어려움  | YAML로 배포 자동화                       |
| 프레임워크마다 다른 API | 공통 REST/gRPC API 제공                |
| 수동 확장          | 자동 확장, 서버리스 지원                     |
| 실험 관리 어려움      | Shadow, A/B 서빙 전략 지원               |
| 멀티모델 환경 비효율    | ModelMesh 기능으로 단일 Pod에 다수 모델 로딩 가능 |

---

## 📚 지원 모델 유형

* `tensorflow`, `pytorch`, `onnx`, `triton`, `sklearn`, `xgboost`, `lightgbm`, `huggingface`, `custom`

---

## ✅ 실전에서 KServe는 이렇게 사용됩니다

| 환경              | 사용 사례                    |
| --------------- | ------------------------ |
| 기업용 API 추론      | REST로 실시간 추론 제공          |
| A/B 테스트 실험      | 두 모델을 동시에 배포하고 트래픽 나눠 보기 |
| 트래픽 없을 때 리소스 절감 | autoscaling to zero 기능   |
| 학습 후 모델 파일 배포   | S3 또는 PVC에 모델 저장 후 자동 배포 |

---

## 💡 KServe vs 대체 도구

| 기능                | KServe      | TorchServe   | Triton    | BentoML   |
| ----------------- | ----------- | ------------ | --------- | --------- |
| 다중 프레임워크          | ✅           | ❌ (PyTorch만) | ✅         | ✅         |
| Kubernetes-native | ✅           | ❌            | ❌         | 🔶        |
| Autoscaling       | ✅ (Knative) | ❌            | ❌         | 🔶        |
| A/B Test, Shadow  | ✅           | ❌            | ❌         | 🔶        |
| REST/gRPC 통합 API  | ✅           | REST만        | REST/gRPC | REST/gRPC |

---

## ✅ 정리 요약

| 항목    | 설명                                                |
| ----- | ------------------------------------------------- |
| 이름    | KServe (구 KFServing)                              |
| 목적    | 모델을 Kubernetes에서 자동으로 서빙                          |
| 주요 기능 | 다중 프레임워크, 자동 스케일링, REST/gRPC, A/B Test            |
| 설치 방식 | Helm 또는 YAML                                      |
| 실습 환경 | Minikube, GKE, EKS, AKS 등 Kubernetes 기반이면 어디서든 OK |

---

