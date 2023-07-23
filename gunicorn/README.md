- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [WSGI](#wsgi)
  - [AWS ELB, NginX, Gunicorn](#aws-elb-nginx-gunicorn)
  - [Gunicorn/Flask Application on Kubernetes](#gunicornflask-application-on-kubernetes)
- [GUnicorn Multiple Workers](#gunicorn-multiple-workers)

-----

# Abstract

**Gunicorn**, which stands for Green Unicorn, is a Python Web Server Gateway Interface (WSGI) HTTP server. It is a pre-fork worker model, meaning it forks multiple worker processes to handle incoming requests, enabling the application to efficiently manage concurrent connections. **Gunicorn** is used to serve Python web applications behind a reverse proxy like Nginx, allowing the application to handle more traffic and manage resources effectively. It is compatible with various web frameworks like Django, Flask, and Pyramid, and is commonly employed in production environments for deploying web applications.

# Materials

* [Nginx와 Gunicorn 둘 중 하나만 써도 될까? | velog](https://velog.io/@jimin_lee/Nginx%EC%99%80-Gunicorn-%EB%91%98-%EC%A4%91-%ED%95%98%EB%82%98%EB%A7%8C-%EC%8D%A8%EB%8F%84-%EB%90%A0%EA%B9%8C)

# Basic

## WSGI

**WSGI**, or **Web Server Gateway Interface**, is a standard interface defined
in **PEP 3333** for connecting Python web applications with web servers. It serves
as a middleware that enables communication between a Python web application or
framework and a web server, allowing for the standardized deployment of Python
web apps across different web servers.

WSGI acts as a bridge between the application and the server, helping to
maintain a clear separation between the application code and the server's
handling of HTTP requests and responses. This separation makes it easier to
develop, deploy, and scale Python web applications across different server
infrastructures. Many popular Python web frameworks, such as Django, Flask, and
Pyramid, conform to the WSGI interface, so they can be easily integrated with
compatible web servers like **Gunicorn** and **uWSGI**.

## AWS ELB, NginX, Gunicorn

Using Nginx along with Gunicorn is not a strict requirement when using AWS
Elastic Load Balancer (ELB) because the load balancer can directly forward the
traffic to your Gunicorn instances. However, there are still a few benefits to
using Nginx as a reverse proxy in conjunction with Gunicorn even when you have
AWS ELB in place:

* **SSL/TLS termination**: While AWS ELB can handle SSL/TLS termination, using Nginx
  allows you to manage certificates and configurations directly within your
  instances. This way, you may find it easier to work with custom configurations
  or when migrating away from AWS infrastructure.
* **Request buffering**: Nginx can act as a buffer between slow clients and your
  application, holding onto the requests until they are fully received, and then
  forwarding them to Gunicorn. This allows Gunicorn workers to focus on processing
  the requests instead of waiting for slow clients, improving overall application
  performance.
* **Static content**: Nginx excels at serving static files from your application
  quickly and efficiently. By delegating this task to Nginx, you reduce the
  workload on your Gunicorn application server, leading to improved performance.
* **Caching**: Nginx provides you with several caching mechanisms, which can be
  leveraged to speed up content delivery and reduce any additional processing by
  Gunicorn.

In conclusion, using Nginx with Gunicorn is still recommended even when using
AWS ELB, because it provides additional benefits that can improve the
performance, security, and flexibility of your application deployment. However,
if your architecture doesn't require these features, you may connect AWS ELB
directly to the Gunicorn instances.

## Gunicorn/Flask Application on Kubernetes

Deploying a **Flask** application with **Gunicorn** in a
[Kubernetes](/kubernetes/README.md) pod is a good approach. This method allows
your application to run in a scalable and distributed Kubernetes environment,
providing the benefit of containerized deployment and the power of Kubernetes to
manage your application.

Here are the main steps to follow when deploying a Flask/Gunicorn application in
a Kubernetes pod:

Create a Docker container image with both Flask and Gunicorn. Use a Dockerfile
to define the base image, dependencies, and configurations for your container.
Make sure to set the entrypoint or command to start your Flask application with
Gunicorn.

Push the created container image to a container registry, such as Docker Hub or
Google Container Registry.

Create a Kubernetes Deployment manifest that defines the pod configuration with
your container image, desired replicas, and other resource and
liveness/readiness probe settings as required.

Expose your application to the network using a Kubernetes Service or Ingress.
This allows access to your Flask application from outside the Kubernetes cluster
and takes advantage of built-in load balancing features.

Apply the Kubernetes Deployment and Service/Ingress manifests to your cluster
using kubectl apply command.

Deploying your Flask/Gunicorn application in a Kubernetes pod will enable you to
make use of Kubernetes' features like autoscaling, rolling updates, and
self-healing, making it easier to manage your application in a distributed
environment.

# GUnicorn Multiple Workers

It is recommended to use multiple worker processes with Gunicorn, particularly
when deploying a production web application. It is correct that **GIL (Global
Interpreter Lock)** in CPython can limit the parallel execution of threads within
a single Python process, but Gunicorn uses a pre-fork worker model, which
manages multiple instances of worker processes, each with their own Python
interpreter, to bypass this limitation.

```
workers = (number_of_cpu_cores * 2) + 1

For example, 1000 millicores, which is equivalent to 1 vCPU:

workers = (1 * 2) + 1 = 3
```
