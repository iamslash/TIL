- [Materials](#materials)
- [Install](#install)
  - [Install with Docker](#install-with-docker)
  - [Install on ubuntu](#install-on-ubuntu)
  - [* How To Install Nginx on Ubuntu 18.04](#ullihow-to-install-nginx-on-ubuntu-1804liul)
- [Architecture](#architecture)
- [Starting, Stopping, Reloading Configuration](#starting-stopping-reloading-configuration)
- [Logs](#logs)
- [Basic](#basic)
  - [Configuration Files' Structure](#configuration-files-structure)
  - [Hello NginX](#hello-nginx)
  - [Server Block](#server-block)
  - [Location Block & Variables](#location-block--variables)
  - [Redirect & Rewrite](#redirect--rewrite)
  - [Try Files & Named Location](#try-files--named-location)
  - [Logging](#logging)
  - [Worker Process](#worker-process)
  - [Buffer & Timeout](#buffer--timeout)
  - [Adding Dynamic Module](#adding-dynamic-module)
  - [Header & Expire](#header--expire)
  - [Compressed Response with gzip](#compressed-response-with-gzip)
  - [FastCGI Cache](#fastcgi-cache)
  - [HTTP/2](#http2)
  - [HTTPS (SSL)](#https-ssl)
  - [Rate Limiting](#rate-limiting)
  - [Basic Auth](#basic-auth)
  - [Hardening Nginx](#hardening-nginx)
  - [Let's Encrypt - SSL Certificates](#lets-encrypt---ssl-certificates)
  - [Reverse Proxy](#reverse-proxy)
  - [Load Balancer](#load-balancer)
  - [Templating with Confd](#templating-with-confd)
  - [Resolver](#resolver)
- [Advanced](#advanced)
  - [Cache setting](#cache-setting)
  - [How to log real-ip](#how-to-log-real-ip)

----

# Materials

* [Nginx @ joinc](https://www.joinc.co.kr/w/man/12/nginx)
* [[Nginx] Overview & Install](https://velog.io/@minholee_93/Nginx-Overview-Install)
  * [Nginx fundamentals @ udemy](https://www.udemy.com/course/nginx-fundamentals/)
  * 킹왕짱 시리즈 
* [Chapter “nginx” in “The Architecture of Open Source Applications”](http://www.aosabook.org/en/nginx.html)
  * nginx internals
* [NGINX Reverse Proxy](https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy/#)
* [proxy_pass](http://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_pass)
* [[Nginx] Configuration](https://velog.io/@minholee_93/Nginx-Configuration-ntk600tiut)
* [Nginx Fundamentals: High Performance Servers from Scratch](https://www.udemy.com/course/nginx-fundamentals/)

# Install

## Install with Docker

* [nginx @ dockerhub](https://hub.docker.com/_/nginx)
* [Beginner’s Guide](http://nginx.org/en/docs/beginners_guide.html)

-----

volume mount 가 제대로 되지 않는 다면 [Nginx is not starting with Docker-compose](https://github.com/nginxinc/docker-nginx/issues/360) 를 참고한다.

```bash
$ docker run --rm --name my-nginx -d -p 80:80 nginx

# copy /etc/nginx directory to /Users/davidsun/my/dockervolume/nginx
$ docker cp my-nginx:/etc/nginx ~/my/dockervolume
$ docker run --rm --name my-nginx -d -p 80:80 nginx
$ docker stop my-nginx

$ docker run --rm --name my-nginx -d -p 80:80 -v /Users/davidsun/my/dockervolume/nginx:/etc/nginx nginx
$ docker exec -it my-nginx bash
> apt-get update
> apt-get install vim
> apt-get install procps
```

## Install on ubuntu

* [[Nginx] Ubuntu에서 apt-get을 통해 Nginx 설치하기 및 간단한 정리](https://twpower.github.io/39-install-nginx-on-ubuntu-by-using-apt-get-and-brief-explanation)
* [How To Install Nginx on Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-install-nginx-on-ubuntu-18-04)
----

```bash
# Install Apache
> sudo apt-get update
> sudo apt-get install nginx

# Adjust the Firewall
> sudo ufw app list
> sudo ufw allow 'Nginx HTTP'
> sudo ufw status

# Check your Web Server
> sudo systemctl status nginx
> hostname -I
> ip addr show eth0 | grep inet | awk '{ print $2; }' | sed 's/\/.*$//'
> sudo apt-get install curl
> curl -4 XXX.XXX.XXX

# Manage the Apache Process
> sudo systemctl stop nginx
> sudo systemctl start nginx
> sudo systemctl restart nginx
> sudo systemctl reload nginx
> sudo systemctl disable nginx
> sudo systemctl enable nginx
```

# Architecture

nginx 는 master process 와 worker process 들로 구성된다.

![](arch.png)

# Starting, Stopping, Reloading Configuration

nginx 의 configuration 은 `/etc/nginx, /usr/local/nginx/conf, /usr/local/etc/nginx` 중 하나에 있다.

nginx 는 master process 와 worker process 들로 구성된다. configuration 를 수정했다면 `> nginx -s reload` 를 실행한다. master process 는 바뀐 config 으로 worker process 들을 다시 띄운다.

```
# Shut down gracefully
> nginx -s quit
# Reload the configuration file
> nginx -s reload
# Reopen log files
> nginx -s reopen 
# Shut down immediately (fast shutdown)
> nginx -s stop
```

# Logs

| directory | description |
|-----------|-------------|
| `/var/log/nginx/access.log` | access logs |
| `/var/log/nginx/error.log` | error logs |

# Basic

## Configuration Files' Structure

nginx 는 module 들이 모여서 동작한다. 그리고 각 module 들은 configuration 의 directive 에 의해 handling 된다. 

또한 directive 는 simple directive 와 block directive 로 나뉘어 진다. 

simple directive 는 key 와 parameters 가 whitespace 를 구분자로 구성되어 있다. 그리고 `;` 로 끝난다.

block directive 는 `{` 로 시작하고 `}` 로 끝난다. block directive 안에 다른 directive 가 존재한다면 바깥 block directive 를 context 라고 한다. 예를 들어 `events, http, server, location` 이 해당한다. [Alphabetical index of directives](http://nginx.org/en/docs/dirindex.html) 에서 모든 directive 를 확인할 수 있다.

`nginx.conf` 의 가장 바깥 context 를 `main` context 라고 한다. `events, http` directive 는 `main` context 에 존재한다. `server` 는 `http` context 에 존재한다. `location` 은 `serve` context 에 존재한다.

## Hello NginX

* [[Nginx] Configuration](https://velog.io/@minholee_93/Nginx-Configuration-ntk600tiut)

----

* `/etc/hosts` 에 `local.iamslash.com` 을 추가한다.
* `> mkdir -p /sites/demo`
* nginx.png 를 `/sites/demo` 에 복사한다.
* 다음과 같이 `/etc/nginx/nginx.conf` 를 작성한다. `events, http` block directive 는 꼭 있어야 한다. `http` 안에 `server` block 을 작성한다.

```conf
events {
}

http {
    include mime.types;
    server {
        listen 80;
        server_name local.iamslash.com;
        root /sites/demo;
    }
}
```

다음과 같이 테스트 한다. `curl http://local.iamslash.com/nginx.png`

## Server Block

* [Server Block Examples @ nginx](https://www.nginx.com/resources/wiki/start/topics/examples/server_blocks/)

## Location Block & Variables

* [[Nginx] Location Block & Variables](https://velog.io/@minholee_93/Nginx-Configuration-2-ask60bxdeh)

----

`location` 은 specific uri 에 대한 behavior 를 정의한다. `server` block 안에 작성한다.

```
events {
}

http {

    include mime.types;

    server {
        listen 80;
        server_name local.iamslash.com;

        root /sites/demo;

        # prefix match
        location /greet {
            return 200 'Hello from NGINX "/greet" location.';
        }
    }
}
```

`/etc/nginx/config` 를 수정하면 `> nginx -s reload` 를 실행해야 한다. 이제 테스트 해보자. `curl http://local.iamslash.com/greet`

uri 가 match 되는 방식은 `prefix match, exact match, regex match, preferential prefix match` 가 있다.

* prefix match
  * greet 으로 시작하는 uri 이 match 된다.

```
# prefix match
location /greet {
    return 200 'this is prefix match';
}
```

* exact match
  * `=` 을 사용하면 `/greet` 만 match 된다.

```
# exact match
location = /greet {
    return 200 'this is exact match';
}
```

* regex match
  * `~` 을 사용하면 case sensitive regext match 가 가능하다.

    ```
    # case sensitive regex match
    location ~ /greet[0-9] {
        return 200 'this is regex match';
    }
    ```
  * `~*` 을 사용하면 case insensitive regext match 가 가능하다.

    ```
    # case insensitive regex match
    location ~* /greet[0-9] {
        return 200 'this is regex match';
    } 
    ```

* preferential prefix match
  * prefix match 와 같다. 그러나 우선순위가 prefix, regext match 보다 높다.

    ```
    # preferential prefix match
    location ^~ /greet {
        return 200 'this is regex match';
    }
    ```
* match order

  ```
  exact > preferential > regex > prefix 
  ```

nginx 는 module variable, configuration variable 을 갖는다. [Alphabetical index of variables](http://nginx.org/en/docs/varindex.html) 에서 nginx 의 모든 variable 들을 확인할 수 있다.

* [module variable](http://nginx.org/en/docs/varindex.html)
  * nginx 의 predefined variable 이다.

    ```
    # nginx module variable : $host / $uri / $args 
    location /inspect {
        return 200 "$host\n$uri\n$args";
    }
    ```

  * `$args` 에서 `name` 을 추출할 수도 있다.
  
    ```
    # return only name args value
    location /inspect {
        return 200 "$args_name";
    }
    ```
* configuration variable

  ```
    server {
        # configuration variable
        set $weekend 'No';

        # check if weekend
        if ( $date_local ~ 'Saturday|Sunday' ){
            set $weekend 'Yes';
        }
        
        # return $weekend value
        location /is_weekend {	
            return 200 $weekend;
        }
    }
  ```

## Redirect & Rewrite

* [[Nginx] Redirect & Rewrite](https://velog.io/@minholee_93/Nginx-Redirect-Rewrite-iwk60eoc6w)

-----

아래와 같이 return 대신 redirect 를 사용하면 redirect 가 가능하다.

```
# redirect
location /logo {
    return 307 /thumb.png;
}
```

다음과 같이 `rewrite` 를 사용하면 url 이 다시 쓰여진다. 즉, client 입장에서는 url 이 바뀌지는 않고 server 에서 처리되는 url 은 달라진다.

```
# rewrite
rewrite ^/logo?$ /thumb.png last;
```

다음은 `^/user/\w+` 를 `/greet` 으로 rewrite 한 예이다.

```
# rewrite
rewrite ^/user/\w+ /greet;

location /greet {
	return 200 "Hello User"
}
```

`/greet/` 다음의 항목을 `$1` 에 mapping 할 수도 있다.

```
rewrite ^/user/(\w+) /greet/$1;

location /greet {
    return 200 "Hello User";
}

location = /greet/minho {
    return 200 "Hello Minho";
}
```

rewrite flag 는 last, break, redirect, permanent 가 있다. [nginx rewrite 플래그의 차이점: last 와 break](https://ohgyun.com/541), [ngx_http_rewrite_module](http://nginx.org/en/docs/http/ngx_http_rewrite_module.html)

* last
  * stops processing the current set of ngx_http_rewrite_module directives and starts a search for a new location matching the changed URI;
* break
  * stops processing the current set of ngx_http_rewrite_module directives as with the break directive;
* redirect
  * returns a temporary redirect with the 302 code; used if a replacement string does not start with “http://”, “https://”, or “$scheme”;
* permanent
  * returns a permanent redirect with the 301 code.

다음과 같은 configuration 를 살펴보고 last, break 를 이해해 보자.

```
A0   location /a {
A1     rewrite ^ /b;
A2     rewrite ^ /c;
A3     add_header x-a a;
A4     proxy_set_header foo foo;
A5     proxy_pass http://localhost:3000;
A6   }

B0   location /b { # B
B1     add_header x-b b;
B2     proxy_pass http://localhost:3000;
B3   }

C0   location /c { # C
C1     rewrite ^ /d;
C2     add_header x-c c;
C3     proxy_pass http://localhost:3000;
C4   }

D0   location /d { # D
D1     add_header x-d d;
D2     proxy_pass http://localhost:3000;
D3   }
```

* `/a` request 가 접수되면 nginx 은 다음과 같은 순서대로 처리한다. 
  * `A0 -> A1 -> A2 -> C0 -> C1 -> D0 -> D1 -> D2`

```
A0   location /a {
A1     rewrite ^ /b last;
A2     rewrite ^ /c;
A3     add_header x-a a;
A4     proxy_set_header foo foo;
A5     proxy_pass http://localhost:3000;
A6   }
```

* 위와 같이 last 를 사용하고 `/a` request 가 접수되면 nginx 은 다음과 같은 순서대로 처리한다. 
  * `A0 -> A1 -> B0 -> B1 -> B2`

```
A0   location /a {
A1     rewrite ^ /b break;
A2     rewrite ^ /c;
A3     add_header x-a a;
A4     proxy_set_header foo foo;
A5     proxy_pass http://localhost:3000;
A6   }
```

* 위와 같이 break 를 사용하고 `/a` request 가 접수되면 nginx 은 다음과 같은 순서대로 처리한다. 
  * `A0 -> A1 -> A3 -> A4 -> A5`

## Try Files & Named Location

* [[Nginx] Try Files & Named Location](https://velog.io/@minholee_93/Nginx-Try-Files-Named-Location-dkk60lodj0)

----

`try files` 은 다음과 같이 `server` 혹은 `location` context 에서 사용한다.

```
server {
    try_files path1 path2 path3
    localtion / {
        try_files path1 path2 path3
    }
}
```

root 에 존재하는 path 를 rewrite 한다???

```
server {

	listen 80;
	server_name 54.180.79.141;

	root /sites/demo;

	try_files /thumb.png /greet;

	location /greet {
        return 200 "Hello User";
	}
}
```

`http://local.iamslash.com/thumb.png` 에 대해 `/greet` 가 존재하므로 `http://local.iamslash.com/greet` 으로 rewrite 한다. 

다음과 같이 `@404` 와 같이  named location 도 가능하다.

```
server {
	listen 80;
	server_name local.iamslash.com;

	root /sites/demo;
	try_files /cat.png /greet @404;
    location @404 {
    	return 404 "Sorry, that file could not be found";
    }

	location /greet {
        return 200 "Hello User";
	}
}
```

## Logging

uri 별로 logging 할 수도 있다. 다음은 `/secure` uri 에대해서 별도의 logging 을 하는 예이다.

```
location /secure {
	access_log /var/log/nginx/secure.access.log;
	return 200 "Welcome to secure area";
}
```

또한 특정 uri 에 대해 logging 하지 않을 수도 있다.

```
location /secure {
	access_log off;
	return 200 "Welcome to secure area";
}
```

## Worker Process

다음과 같이 worker process 의 수를 변경할 수 있다.

```
worker_processes 2;
events {}
http {
    include mime.types;
    server {
        listen 80;
        server_name local.iamslash.com;
        root /sites/demo;
    }
}
```

다음과 같이 pid 의 경로를 바꿀 수도 있다. 또한 worker process 의 connection 수자를 변경할 수도 있다. `ulimit -n` 을 실행하면 max file open number 를 확인할 수 있다.

```
worker_processes auto;
pid /run/new_nginx.pid;

events {
	worker_connections 1024;
}
http {
    include mime.types;

    server {
        listen 80;
        server_name local.iamslash.com;
        root /sites/demo;
    }
}
```

## Buffer & Timeout

다음은 Buffer configuration 의 모음이다. 각 directive 의 parameter 로 `100, 100k, 100M` 등이 가능하다. 

| directive example | description |
|---|---|
| `client_body_buffer_size 10K;` | form submission 의 buffer size |
| `client_max_body_size 8m;` | form submission 의 max body size. 8m 을 넘으면 413 error |
| `client_header_buffer_size 1K;` | client header buffer size |
| `sendfile on;` | DISK 에 저장된 static 파일을 buffer 에 저장하지 않고 바로 보낸다. |
| `tcp_nopush on;` | sendfile 의 packet 을 optimize 한다. |

다음은 Timeout configuration 의 모음이다. 각 directive 의 parameter 로 `10, 10s, 10m, 10h, 10d` 등이 가능하다. 단위 접미사를 생략하면 millisecond 이다.

| directive example | description |
|---|---|
| `client_body_timeout 12;` | 연속적인 읽기작업 사이의 시간제한 |
| `client_header_timeout 12;` | |
| `keepalive_timeout 15;` | 다음 데이터를 받기위해 유지하는 connection timeout |
| `send_timeout 10;` | client 입장에서 server 로 부터 아무런 데이터도 받지 않을 때 connection timeout |

## Adding Dynamic Module

## Header & Expire

## Compressed Response with gzip
## FastCGI Cache
## HTTP/2
## HTTPS (SSL)
## Rate Limiting
## Basic Auth
## Hardening Nginx
## Let's Encrypt - SSL Certificates
## Reverse Proxy

다음과 같이 `http://local.iamslash.com/php` 에 대해 `http://localhost:9999` 로 forwarding 해보자. proxy_pass 의 parameter 는 반드시 `/` 로 끝나야 한다. 만약 uri 가 `http://local.iamslash.com/php/hello` 라면 `http://localhost:9999/hello` 로 forwarding 된다.

```
events {}
http {
    server {
        listen 80;
        location / {
                return 200 "Hello From NGINX\n";
        }
        # set reverse proxy
        location /php {
            # proxy pass uri
            proxy_pass http://localhost:9999/;
        }
    }
}
```

또한 다음과 같이 Custom header `proxied nginx` 를 전달할 수 도 있다.

```
events {}
http {
    server {
        listen 80;
        location / {
                return 200 "Hello From NGINX\n";
        }
        # set reverse proxy
        location /php {
            # proxy header
            proxy_set_header proxied nginx;
            # proxy pass uri
            proxy_pass http://localhost:9999/;
        }
    }
}
```

## Load Balancer

다음과 같이 upstream context 안에 server 들을 구성할 수 있다.

```
events {}
http {
    upstream php_servers {
        # set load balancer servers
        server localhost:10001;
        server localhost:10002;
        server localhost:10003;
    }
    server {
        listen 80;
        location / {
            # proxy pass to load balancer
            proxy_pass http://php_servers;
        }
    }
}
```

다음과 같이 테스트 해본다.

```bash
$ while sleep 0.5; do curl http://local.iamslash.com/; done
```

load balancing rule 은 Sticky Sessions 와 Least Number of Active 방식이 있다. 기본적으로 Round Robin 이다.

아래와 같이 `/etc/nginx/nginx.conf` 를 수정하면 `localhost:10001` 으로만 request 가 forwarding 된다.

```
events {}

http {
	upstream php_servers {
        # use sticky session
        ip_hash;
        # set load balancer servers
        server localhost:10001;
        server localhost:10002;
        server localhost:10003;
    }
    server {
        listen 8888;
        location / {
            # proxy pass to load balancer
            proxy_pass http://php_servers;
        }
    }
}
```

다음과 같이 수정하면 가장 connection 이 적은 server 로 forwarding 한다.

```
events {}

http {
	upstream php_servers {
        # use least connection
        least_conn;
        # set load balancer servers
        server localhost:10001;
        server localhost:10002;
        server localhost:10003;
    }

    server {
        listen 8888;
        location / {
            # proxy pass to load balancer
            proxy_pass http://php_servers;
        }
    }
}
```


## Templating with Confd

* [confd](/confd/README.md)

----

confd 와 함께 사용하면 configuration 을 환경변수 등등을 templating 할 수 있다.

```
    location / {
        ...
        client_max_body_size 10m;
        client_body_buffer_size 512k;

        {{ getenv "IAMSLASH_ASSET" }}
    }
```

## Resolver

* [Nginx resolver explained](https://distinctplace.com/2017/04/19/nginx-resolver-explained/)
* [Nginx를 ELB Reverse Proxy로 사용할때 주의 점](http://tech.kkung.net/blog/nginx-with-elb/)

----

resolver 는 DNS server 이다. `172.16.0.23` 은 AWS DNS server 이다. Nginx 는 configuration 을 읽을 때 DNS에 대한 IP 변환(resolve)를 수행한다. 

다음과 같이 resolver 를 설장하고 `proxy_pass` 에 config variable 을 사용하면 runtime 에 `@ep` 의 real ip 를 resolver 를 통해서 얻어온다.

따라서 upstream server 의 IP 가 바뀌어도 configuration 을 reloading 하지 않고 IP 를 resolving 할 수 있다.

```
resolver 172.16.0.23 valid=5s;

set $ep "http://elb-test.ap-northeast-1.elb.amazonaws.com";
location @beat-api {
    proxy_pass http://$ep;
    proxy_set_header   Host $host;
    proxy_set_header   X-Real-IP $remote_addr;
    proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
}
```

# Advanced

## Cache setting

* [[Nginx] 설정 Cache 기능 정리](https://blog.kjslab.com/175)
* [NginX를 이용한 static 컨텐츠 서비스](https://www.joinc.co.kr/w/man/12/nginx/static)

------

예를 들어 S3 에 static contents 를 저장하고 nginx 로 caching 할 수 있다.

## How to log real-ip

* [Proxy(프락시) 환경에서 client IP 를 얻기 위한 X-Forwarded-For(XFF) http header](https://www.lesstif.com/pages/viewpage.action?pageId=20775886)
* [How to log real user’s IP address with Nginx in log files](https://www.cyberciti.biz/faq/linux-unix-nginx-access-file-log-real-ip-address-of-visitor/)

