# Materials

* [[Nginx] Overview & Install](https://velog.io/@minholee_93/Nginx-Overview-Install)
  * [Nginx fundamentals @ udemy](https://www.udemy.com/course/nginx-fundamentals/)
  * 킹왕짱 시리즈 
* [Proxy에 대하여 @ joinc](https://www.joinc.co.kr/w/man/12/proxy)
* [NGINX Reverse Proxy](https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy/#)
* [proxy_pass](http://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_pass)

# Install

## Install with Docker

* [nginx @ dockerhub](https://hub.docker.com/_/nginx)

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

다음은 nginx 의 command 이다. config 를 수정했다면 `> nginx -s reload` 를 실행한다.

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

# Configuration Files

| directory | description |
|-----------|-------------|
| `/var/www/html` | The Nginx configuration directory.  |
| `/etc/nginx/nginx.conf` | The main Nginx configuration file.  |
| `/etc/nginx/sites-available/` |  The directory where per-site server blocks can be stored. |
| `/etc/nginx/sites-enabled/` | The directory where enabled per-site server blocks are stored.  |
| `/etc/nginx/snippets` | This directory contains configuration fragments that can be included elsewhere in the Nginx configuration. |

# Logs

| directory | description |
|-----------|-------------|
| `/var/log/nginx/access.log` | access logs |
| `/var/log/nginx/error.log` | error logs |

# Basic

## Hello NginX

* [[Nginx] Configuration](https://velog.io/@minholee_93/Nginx-Configuration-ntk600tiut)

----

* `/etc/hosts` 에 `local.iamslash.com` 을 추가한다.
* `> mkdir -p /sites/demo`
* nginx.png 를 /sites/demo 에 복사한다.
* 다음과 같이 `/etc/nginx/nginx.conf` 를 작성한다. event, http block 은 꼭 있어야 한다. http block 안에 server block 을 작성한다.

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

* 이제 `http://local.iamslash.com/nginx.png` 를 접속한다.

## Location Block & Variables

* [[Nginx] Location Block & Variables](https://velog.io/@minholee_93/Nginx-Configuration-2-ask60bxdeh)

----

location은 specific uri 에 대한 behavior 를 정의한다. server block 안에 작성한다.

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

config 를 수정하면 `> nginx -s reload` 를 실행해야 한다.
이제 browser 에서 `http://local.iamslash.com/greet` 에 접근해 본다.

uri 가 match 되는 방식은 `prefix match, exact match, regex match, preferential prefix match` 가 있다.

* prefix match
  * greet 으로 시작하는 url 이 match 된다.

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

nginx 는 module variable, configuration variable 을 갖는다. 

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

다음과 같이 `rewrite` 를 사용하면 url 이 다시 쓰여진다. 그러나 client 입장에서는 url 이 바뀌지는 않는다.

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


## Try Files & Named Location

* [[Nginx] Try Files & Named Location](https://velog.io/@minholee_93/Nginx-Try-Files-Named-Location-dkk60lodj0)

----

`try files` 은 다음과 같이 server & location context 에서 사용한다.

```
server {
    try_files path1 path2 path3
    localtion / {
        try_files path1 path2 path3
    }
}
```

## Logging

## Worker Process

## Buffer & Timeout

## Adding Dynamic Module

## Header & Expire

## Compressed Response with gzip
## GastCGI Cache
## HTTP/2
## HTTPS (SSL)
## Rate Limiting
## Basic Auth
## Hardening Nginx
## Let's Encrypt - SSL Certificates
## Reverse Proxy
## Load Balancer
