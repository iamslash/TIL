# Install on ubuntu

* [Apache2 설치 (Ubuntu 16.04)](https://lng1982.tistory.com/288)
* [How To Install the Apache Web Server on Ubuntu 16.04 @ digitalocean](https://www.digitalocean.com/community/tutorials/how-to-install-the-apache-web-server-on-ubuntu-16-04)
* [How To Set Up Apache Virtual Hosts on Ubuntu 16.04 @ digitalocean](https://www.digitalocean.com/community/tutorials/how-to-set-up-apache-virtual-hosts-on-ubuntu-16-04)

----

```bash
# Install Apache
> sudo apt-get update
> sudo apt-get install apache2

# Adjust the Firewall
> sudo ufw app list
> sudo ufw allow 'Apache Full'
> sudo ufw status

# Check your Web Server
> sudo systemctl status apache2
> hostname -I
> sudo apt-get install curl
> curl -4 XXX.XXX.XXX

# Manage the Apache Process
> sudo systemctl stop apache2
> sudo systemctl start apache2
> sudo systemctl restart apache2
> sudo systemctl reload apache2
> sudo systemctl disable apache2
> sudo systemctl enable apache2
```

# Configuration

| directory | description |
|-----------|-------------|
| `/var/www/html` | content |
| `/etc/apache2` | he Apache configuration directory.  |
| `/etc/apache2/apache2.conf` | The main Apache configuration file |
| `/etc/apache2/ports.conf` | This file specifies the ports that Apache will listen on |
| `/etc/apache2/sites-available/` | The directory where per-site "Virtual Hosts" can be stored.  |
| `/etc/apache2/sites-enabled/` | The directory where enabled per-site "Virtual Hosts" are stored.  |
| `/etc/apache2/conf-available/, /etc/apache2/conf-enabled/` | These directories have the same relationship as the sites-available and sites-enabled directories, |
| `/etc/apache2/mods-available/, /etc/apache2/mods-enabled/` | These directories contain the available and enabled modules, respectively.  |

# Logs

| directory | description |
|-----------|-------------|
| `/var/log/apache2/access.log` | access logs |
| `/var/log/apache2/error.log` | error logs |

