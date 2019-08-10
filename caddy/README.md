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
```