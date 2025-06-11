
# Install Triton On K9s

minikube, kserve, istio, knative, Triton 을 설치해 본다.

```bash
#!/bin/bash

set -e

echo "🚀 Minikube 클러스터 삭제 중..."
minikube delete

echo "🚀 Minikube 클러스터 시작 중..."
minikube start --memory=8192 --cpus=4 --kubernetes-version=v1.27.3 --driver=docker

echo "✅ Minikube 클러스터 시작 완료"

echo "📦 Istio 설치 중..."
kubectl create namespace istio-system || true
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.11.0/istio.yaml

echo "⏳ Istio 구성 요소 준비 중..."
kubectl wait --for=condition=Available deployment/istiod -n istio-system --timeout=180s
kubectl wait --for=condition=Available deployment/istio-ingressgateway -n istio-system --timeout=180s
echo "✅ Istio 설치 완료"

echo "📦 Knative Serving 설치 중..."
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.11.0/serving-crds.yaml
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.11.0/serving-core.yaml

echo "⏳ Knative 구성 요소 준비 중..."
kubectl wait --for=condition=Available deployment/controller -n knative-serving --timeout=180s
kubectl wait --for=condition=Available deployment/webhook -n knative-serving --timeout=180s
echo "✅ Knative Serving 설치 완료"

echo "🔗 Istio와 Knative 연동 중..."
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.11.0/net-istio.yaml

echo "📦 cert-manager 설치 중..."
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.4/cert-manager.yaml

echo "📦 KServe 설치 중..."
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.2/kserve.yaml

echo "⏳ KServe 구성 요소 준비 중..."
kubectl wait --for=condition=Available deployment/kserve-controller-manager -n kserve --timeout=180s
echo "✅ KServe 설치 완료"

echo "✅ 전체 설치 완료! 이제 InferenceService를 배포하고 KServe를 사용할 수 있습니다."

# Knative Gateway 동작 확인 (포트포워딩)
kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80
```

# Install Triton Model

```bash
# Mount on minikube
minikube mount /Users/david.s/prj/github/iamslash/ml-ex/triton-ex/basic/triton-models:/mnt/models

kubectl create namespace demo || true
kubectl label namespace demo istio-injection=enabled
kubectl label namespace demo serving.kserve.io/inferenceservice=enabled

kubectl apply -f pvc-models.yaml
kubectl apply -f linear-regression-pvc.yaml
```

```yml
-- pvc-linear-model.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-pv
spec:
  capacity:
    storage: 1Gi
  accessModes:
    - ReadOnlyMany
  hostPath:
    path: "/mnt/models"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
  namespace: demo
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 1Gi
  volumeName: model-pv

```

```yml
-- linear-regression-pvc.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: linear-regression-v2
  namespace: demo
spec:
  predictor:
    triton:
      storageUri: "pvc://model-pvc/linear_regression"
```
