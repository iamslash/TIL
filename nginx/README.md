# Materials

* [Proxy에 대하여 @ joinc](https://www.joinc.co.kr/w/man/12/proxy)
* [NGINX Reverse Proxy](https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy/#)
* [proxy_pass](http://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_pass)

# Install on ubuntu

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

# Setting Up Server Blocks 

```bash
> sudo mkdir -p /var/www/example.com/html
> sudo chown -R $USER:$USER /var/www/example.com/html
> sudo chmod -R 755 /var/www/example.com
> nano /var/www/example.com/html/index.html

<html>
    <head>
        <title>Welcome to Example.com!</title>
    </head>
    <body>
        <h1>Success!  The example.com server block is working!</h1>
    </body>
</html>

> sudo nano /etc/nginx/sites-available/example.com

server {
        listen 80;
        listen [::]:80;

        root /var/www/example.com/html;
        index index.html index.htm index.nginx-debian.html;

        server_name example.com www.example.com;

        location / {
                try_files $uri $uri/ =404;
        }
}

> sudo ln -s /etc/nginx/sites-available/example.com /etc/nginx/sites-enabled/
> sudo nano /etc/nginx/nginx.conf

...
http {
    ...
    server_names_hash_bucket_size 64;
    ...
}
...

> sudo nginx -t
> sudo systemctl restart nginx
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

