
## Rolling updates and Rollbacks

```bash
$ kubectl create -f deployment.yml
$ kubectl get deployments
$ kubectl apply -f deployment-definition.yml
$ kubectl set image deployment/myapp-deployment nginx=nginx:1.9.1
$ kubectl rollout status deployment/myapp-deployment
$ kubectl rollout history deployment/myapp-deployment
$ kubectl rollout undo deployment/myapp-deployment
```
## Commands

```bash
$ docker build -t ubuntu-sleeper .
$ docker run ubuntu-sleeper
$ docker run ubuntu-sleeper sleep 10
$ docker run ubuntu-sleeper 10
$ docker run --entrypoint sleep unbuntu-sleeper 10
```

## Commands and Arguments

```bash
$ kubectl create -f pod-definition.yml
```

* `pod-definition.yml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: ubuntu-sleeper-pod
spec:
  containers:
    - name: ubuntu-sleeper
      image: ubuntu-sleeper
      command: ["sleep"]
      args: ["10"]
```

## Configuring ConfigMaps in Applications

```bash
$ docker run -e APP_COLOR=blue simple-webapp-color
$ kubectl create -f pod-definition.yml
```

* `pod-definition.yml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: simple-webapp-color
spec:
  containers:
  - name: simple-webapp-color
    image: simple-webapp-color
    ports:
      - containerPort: 8080
    env:
      - name: APP_COLOR
        value: blue
```

* `env value types`

```yml
env:
  - name: APP_COLOR
    value: pink

env:
  - name: APP_COLOR
    valueFrom:
      configMapKeyRef:
      
env:
  - name: APP_COLOR
    valueFrom:
      secretKeyRef:      
```

```bash
# Imperative
# kubectl create configmap
#    <config-name> --from-literal=<key>=<value>
$ kubectl create configmap
    app-config --from-literal=APP_COLOR=blue \
               --from-literal=APP_MOD=prod

# Declarative
# kubel create configmap
    <config-name> --from-file=<path-to-file>
$ kubel create configmap
    <config-name> --from-file=app_config.properties
```

* `app_config.properties`

```conf
APP_COLOR: blue
APP_MODE: prod
```

* `pod-definition.yml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: simple-webapp-color
spec:
  containers:
  - name: simple-webapp-color
    image: simple-webapp-color
    ports:
      - containerPort: 8080
    envFrom:
    - configMapRef:
        name: app-config
```

```bash
$ kubectl create -f config-map.yaml

$ kubectl get configmaps

$ kubectl describe configmaps
```

* `config-map.yml`

```yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  APP_COLOR: blue
  APP_MODE: prod  
```

## Configure Secrets in Applications

## A note about Secrets!

## Scale Applications

## Multi Container PODs

## Multi-Container PODs Design Paterns

## InitContainers

## Self Healing Applications