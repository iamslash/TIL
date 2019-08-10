# Materials

* [How To Host a Website with Caddy on Ubuntu 16.04](https://www.digitalocean.com/community/tutorials/how-to-host-a-website-with-caddy-on-ubuntu-16-04)

# Install on Ubuntu

```bash
# Building Caddy
> go get github.com/mholt/caddy/caddy
> cd $GOPATH/src/github.com/mholt/caddy
> git tag
> git checkout -b "adding_plugins" "v0.10.12"
> go install github.com/mholt/caddy/caddy
> caddy
> nano $GOPATH/src/github.com/mholt/caddy/caddy/caddymain/run.

. . .
// Run is Caddy's main() function.
func Run() {
        fmt.Println("Hello from Caddy!")

        flag.Parse()

        caddy.AppName = appName
        . . .
}

> go install github.com/mholt/caddy/caddy 
> caddy
```

```bash
# Installing Caddy
> sudo cp $GOPATH/bin/caddy /usr/local/bin/
> sudo chown root:root /usr/local/bin/caddy
> sudo chmod 755 /usr/local/bin/caddy
# Using the setcap command can allow the Caddy process to bind to low ports without running as root
> sudo setcap 'cap_net_bind_service=+ep' /usr/local/bin/caddy
> sudo mkdir /etc/caddy
> sudo chown -R root:www-data /etc/caddy
> sudo mkdir /etc/ssl/caddy
> sudo chown -R root:www-data /etc/ssl/caddy
> sudo chmod 0770 /etc/ssl/caddy
> sudo mkdir /var/www
> sudo chown www-data:www-data /var/www
> sudo touch /etc/caddy/Caddyfile
> sudo cp $GOPATH/src/github.com/mholt/caddy/dist/init/linux-systemd/caddy.service /etc/systemd/system/
> sudo chmod 644 /etc/systemd/system/caddy.service
> sudo systemctl daemon-reload
> sudo systemctl status caddy
> sudo ufw allow 80
> sudo ufw allow 443
> sudo ufw status

# Configuring Caddy
> sudo touch /var/www/index.html
> sudo nano /var/www/index.html

<!DOCTYPE html>
<html>
  <head>
    <title>Hello from Caddy!</title>
  </head>
  <body>
    <h1 style="font-family: sans-serif">This page is being served via Caddy</h1>
  </body>
</html>

> sudo nano /etc/caddy/Caddyfile

:80 {
    root /var/www
    gzip {
        ext .html .htm .php
        level 6
    }
}

> sudo systemctl start caddy
> sudo systemctl status caddy
> sudo systemctl stop caddy

# Using Plugins

> cd $GOPATH/src/github.com/mholt/caddy
> nano caddy/caddymain/run.go

. . .
import (
    "errors"
    "flag"
    "fmt"
    "io/ioutil"
    "log"
    "os"
    "runtime"
    "strconv"
    "strings"

    "gopkg.in/natefinch/lumberjack.v2"

    "github.com/xenolf/lego/acmev2"

    "github.com/mholt/caddy"
    // plug in the HTTP server type
    _ "github.com/mholt/caddy/caddyhttp"

    "github.com/mholt/caddy/caddytls"
    // This is where other plugins get plugged in (imported)
)
. . .

> nano caddy/caddymain/run.go

. . .
import (
    . . .
    "github.com/mholt/caddy/caddytls"
    // This is where other plugins get plugged in (imported)

    _ "github.com/hacdias/caddy-minify"
)

> git config --global user.email "sammy@example.com"
> git config --global user.name "Sammy"
> git add -A .
> git commit -m "Added minify plugin"
> go get ./...
> go install github.com/mholt/caddy/caddy
> sudo cp $GOPATH/bin/caddy /usr/local/bin/
> sudo chown root:root /usr/local/bin/caddy
> sudo chmod 755 /usr/local/bin/caddy
> sudo setcap 'cap_net_bind_service=+ep' /usr/local/bin/caddy
> sudo nano /etc/caddy/Caddyfile

:80 {
    root /var/www
    gzip
    minify
}

> sudo systemctl start caddy

# curl http://example.com
```

```bash

# Enabling Automatic TLS with Let's Encrypt

> nano $GOPATH/src/github.com/mholt/caddy/caddy/caddymain/run.go
 
. . .
import (
    . . .
    "github.com/mholt/caddy/caddytls"
    // This is where other plugins get plugged in (imported)

    _ "github.com/hacdias/caddy-minify"
    _ "github.com/caddyserver/dnsproviders/digitalocean"
)

> cd $GOPATH/src/github.com/mholt/caddy
> git add -A .
> git commit -m "Add DigitalOcean DNS provider"
> go get ./...
> go install github.com/mholt/caddy/caddy
> sudo systemctl stop caddy
> sudo cp $GOPATH/bin/caddy /usr/local/bin/
> sudo chown root:root /usr/local/bin/caddy
> sudo chmod 755 /usr/local/bin/caddy
> sudo setcap 'cap_net_bind_service=+ep' /usr/local/bin/caddy
> sudo nano /etc/systemd/system/caddy.service

[Service]
Restart=on-abnormal

; User and group the process will run as.
User=www-data
Group=www-data

; Letsencrypt-issued certificates will be written to this directory.
Environment=CADDYPATH=/etc/ssl/caddy DO_AUTH_TOKEN=your_token_here

> sudo systemctl daemon-reload
> sudo systemctl status caddy
> sudo nano /etc/caddy/Caddyfile

example.com {
    root /var/www
    gzip
    minify
    tls {
        dns digitalocean
    }
}

> sudo systemctl start caddy
> sudo systemctl enable caddy

# Updating Your Caddy Installation

> cd $GOPATH/src/github.com/mholt/caddy
> git checkout adding_plugins
> git fetch origin
> git tag
> git merge adding_plugins v0.10.13
> go install github.com/mholt/caddy/caddy
> sudo systemctl stop caddy
> sudo cp $GOPATH/bin/caddy /usr/local/bin/
> sudo chown root:root /usr/local/bin/caddy
> sudo chmod 755 /usr/local/bin/caddy
> sudo setcap 'cap_net_bind_service=+ep' /usr/local/bin/caddy
> sudo systemctl start caddy
```