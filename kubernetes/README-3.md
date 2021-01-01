- [Logging & Monitoring](#logging--monitoring)
  - [Monitor Cluster Components](#monitor-cluster-components)
  - [Managing Application Logs](#managing-application-logs)
- [Application Lifecycle Management](#application-lifecycle-management)
  - [Rolling updates and Rollbacks](#rolling-updates-and-rollbacks)
  - [Configure Applications](#configure-applications)
  - [Commands](#commands)
  - [Commands and Arguments](#commands-and-arguments)
  - [Configure Environment Variables in Applications](#configure-environment-variables-in-applications)
  - [Configuring ConfigMaps in Applications](#configuring-configmaps-in-applications)
  - [Configure Secrets in Applications](#configure-secrets-in-applications)
  - [A note about Secrets!](#a-note-about-secrets)
  - [Scale Applications](#scale-applications)
  - [Multi Container PODs](#multi-container-pods)
  - [Multi-Container PODs Design Paterns](#multi-container-pods-design-paterns)
  - [InitContainers](#initcontainers)
  - [Self Healing Applications](#self-healing-applications)
- [Cluster Maintenance](#cluster-maintenance)
  - [OS Upgrades](#os-upgrades)
  - [Kubenetes Software Versions](#kubenetes-software-versions)
  - [Cluster Upgrade Process](#cluster-upgrade-process)
  - [Backup and Restore Methods](#backup-and-restore-methods)
  - [Working with ETCDCTL](#working-with-etcdctl)
- [Security](#security)
  - [Kubernetes Security Primitives](#kubernetes-security-primitives)
  - [Authentication](#authentication)
  - [Article on Setting up Basic Authentication](#article-on-setting-up-basic-authentication)
  - [A note on Service Accounts](#a-note-on-service-accounts)
  - [TLS Introduction](#tls-introduction)
  - [TLS Basics](#tls-basics)
  - [TLS in Kubernetes](#tls-in-kubernetes)
  - [TLS in Kubernetes - Certificate Creation](#tls-in-kubernetes---certificate-creation)
  - [View Certificate Details](#view-certificate-details)
  - [Certificate API](#certificate-api)
  - [KubeConfig](#kubeconfig)
  - [Persistent Key/Value Store](#persistent-keyvalue-store)
  - [API Groups](#api-groups)
  - [Authorization](#authorization)
  - [Role Based Access Controls](#role-based-access-controls)
  - [Cluster Roles and Role Bindings](#cluster-roles-and-role-bindings)
  - [Image Security](#image-security)
  - [Security Contexts](#security-contexts)
  - [Network Policty](#network-policty)
- [Storage](#storage)
  - [Storage in Docker](#storage-in-docker)
  - [Volume Driver Plugins in Docker](#volume-driver-plugins-in-docker)
  - [Volumes](#volumes)
  - [Persistent Volumes](#persistent-volumes)
  - [Persistent Volume Claims](#persistent-volume-claims)
  - [Using PVCs in PODs](#using-pvcs-in-pods)
  - [Application Configuration](#application-configuration)
  - [Storage Class](#storage-class)
- [Networking](#networking)
- [Design and Install a Kubernetes Cluster](#design-and-install-a-kubernetes-cluster)
- [Install "Kubernetes the kubeadm way"](#install-kubernetes-the-kubeadm-way)
- [End to End Tests on a Kubernetes Cluster](#end-to-end-tests-on-a-kubernetes-cluster)
- [Troubleshooting](#troubleshooting)
- [Other Topics](#other-topics)
- [Lightning Labs](#lightning-labs)
- [Mock Exams](#mock-exams)

-----

# Logging & Monitoring

## Monitor Cluster Components

Metrics Server

```bash
$ minikube addons enable metrics-server

$ git clone https://github.com/kubernetes-incubator/metrics-server.git
$ kubectl create -f deploy/1.8+/
$ kubectl top node
$ kubectl top pod
```

## Managing Application Logs

* `event-sumulator.yaml` Single Container

```yaml
apiVersion: v1
kind: Pod
metadata:
name: event-simulator-pod
spec:
containers:
- name: event-simulator
image: kodekloud/event-simulator
```

```bash
# logs of docker container
$ docker run -d kodekloud/event-simulator
$ docker logs -f ecf

# logs of kubernetes container
$ kubectl create -f event-simulator.yaml
$ kubectl logs -f event-simulator-pod
```

* `event-simulator.yaml` multiple containers

```yaml
apiVersion: v1
kind: Pod
metadata:
name: event-simulator-pod
spec:
containers:
- name: event-simulator
image: kodekloud/event-simulator
event-simulator.yaml
- name: image-processor
image: some-image-processor
```

```bash
$ kubectl logs -f event-simulator-pod event-simulator
```

# Application Lifecycle Management

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

## Configure Applications

## Commands

## Commands and Arguments

## Configure Environment Variables in Applications

## Configuring ConfigMaps in Applications

## Configure Secrets in Applications

## A note about Secrets!

## Scale Applications

## Multi Container PODs

## Multi-Container PODs Design Paterns

## InitContainers

## Self Healing Applications

# Cluster Maintenance

## OS Upgrades

## Kubenetes Software Versions

## Cluster Upgrade Process

## Backup and Restore Methods

## Working with ETCDCTL

# Security

## Kubernetes Security Primitives

## Authentication

## Article on Setting up Basic Authentication

## A note on Service Accounts

## TLS Introduction

## TLS Basics

## TLS in Kubernetes

## TLS in Kubernetes - Certificate Creation

## View Certificate Details

## Certificate API

## KubeConfig

## Persistent Key/Value Store

## API Groups

## Authorization

## Role Based Access Controls

## Cluster Roles and Role Bindings

## Image Security

## Security Contexts

## Network Policty

# Storage

## Storage in Docker

## Volume Driver Plugins in Docker

## Volumes

## Persistent Volumes

## Persistent Volume Claims

## Using PVCs in PODs

## Application Configuration

## Storage Class

# Networking

# Design and Install a Kubernetes Cluster

# Install "Kubernetes the kubeadm way"

# End to End Tests on a Kubernetes Cluster

# Troubleshooting

# Other Topics

# Lightning Labs

# Mock Exams
