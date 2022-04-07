- [Abstract](#abstract)
- [Headless Service](#headless-service)

----

# Abstract

Headless Service 에 대해 정리한다. 

# Headless Service

* [헤드리스(Headless) 서비스](https://kubernetes.io/ko/docs/concepts/services-networking/service/#%ED%97%A4%EB%93%9C%EB%A6%AC%EC%8A%A4-headless-%EC%84%9C%EB%B9%84%EC%8A%A4)
* [Headless Service를 이용하여 네임스페이스가 다른 서비스에 Ingress 연결하기](https://coffeewhale.com/kubernetes/service/2020/01/22/headless-svc/)

-----

때때로 로드-밸런싱과 단일 서비스 IP는 필요치 않다. 이 경우, "헤드리스" 서비스라는 것을 만들 수 있는데, 명시적으로 클러스터 `IP (.spec.clusterIP)`에 "None"을 지정한다.

주로 StatefulSet Pod 를 로드-밸런싱하지 않고 싶을 때 적용한다.

```yaml
# nginx-statefulset.yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx
  labels:
    app: nginx
spec:
  ports:
  - port: 80
    name: web
  clusterIP: None
  selector:
    app: nginx
---

apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: web
spec:
  selector:
    matchLabels:
      app: nginx
  serviceName: "nginx" # headless service name 
  replicas: 3 
  template:
    metadata:
      labels:
        app: nginx
    spec:
      terminationGracePeriodSeconds: 10 
      containers:
      - name: nginx
        image: k8s.gcr.io/nginx-slim:0.8
        ports:
        - containerPort: 80
          name: web
        volumeMounts:
        - name: www
          mountPath: /usr/share/nginx/html
  volumeClaimTemplates: 
  - metadata:
      name: www
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: "standard"
      resources:
        requests:
          storage: 1Gi
```
