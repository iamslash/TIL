# Prerequisite - SwitchingRouting

* [ip Command Cheat Sheet for Red Hat Enterprise Linux](https://access.redhat.com/articles/ip-command-cheat-sheet)

----

Router makes it possible route packets between different networks. Kernel has route tables.

```bash
# Manage and display the state of all network interfaces
$ ip link

# Display IP Addresses and propertey information (abbreviation of address)
$ ip addr

# Add address 192.168.1.10 netmask 24 to device eth0
$ ip addr add 192.168.1.10/24 dev eth0

# Same with > route
# List all of the route entries in the kernel
$ ip route

# Add a route to 192.168.1.0/24 via the gateway at 192.168.2.1
# Default is same with 0.0.0.0 which means anywhere
# If you want to apply this permanently, you need to save this /etc/network/interfaces file
$ ip route add 192.168.1.10/24 via 192.168.2.1

# if ip_forward is 1, it's possible to forward packets in different networks
# If you want to apply this permanently, you need set net.ipv4.ip_forward = 1 in /etc/sysctl.conf
$ cat /prod/sys/net/ipv4/ip_forward
```

# Prerequisite - DNS

We use `/etc/hosts` file to query ip for the name without DNS server.

```bash
192.168.1.11  db
192.168.1.11  www.iamslash.com
```

We use `/etc/resolv.conf` file to query ip for the DNS name with DNS server. It includes DNS server list.

```bash
nameserver  192.168.1.100
```

Host find the ip from `/etc/hosts/` and DNS server in order. But the order could be changed by `/etc/nsswitch.conf`.

```bash
host:   files dns
```

Domain names are consisted of many components such as `www, google, com`. Each components are handled each DNS servers.

```
Client ->  Org  ->  Root  ->  com  ->  Google
           DNS      DNS       DNS      DNS
```

We can use `search` entry for Domain Names qeury in `/etc/resolv.conf`.

```bash
nameserver  192.168.1.100
search      iamslash.com prod.iamslash.com
```

```bash
$ ping web
PING web.iamslash.com ...

$ ping web.iamslash.com
PING web.iamslash.com
```

There are several record types in DNS server. `A` means IP4's ip, `AAAA` means IP6's ip, `CNAME` means other Domain Names.

| Type | Name | Value |
|--|--|--|
| A | web | 192.168.1.1 |
| AAAA | web | 2001:0xx8:85a3:0000:0000:8a2e:0370:73334 |
| CNAME | food.web | eat.web, hungry.web |

You can DNS query with `nslookup`. This does not query from `/etc/hosts`.

```console
$ nslookup www.google.com
Server:         192.168.65.1
Address:        192.168.65.1#53

Non-authoritative answer:
Name:   www.google.com
Address: 172.217.175.100
Name:   www.google.com
Address: 2404:6800:4004:80c::2004
```

You can DNS query with `dig`. This will explain in detail and does not query from `/etc/hosts`.

```console
$ dig www.google.com

; <<>> DiG 9.11.3-1ubuntu1.13-Ubuntu <<>> www.google.com
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 41917
;; flags: qr rd ra; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 0

;; QUESTION SECTION:
;www.google.com.                        IN      A

;; ANSWER SECTION:
www.google.com.         266     IN      A       172.217.175.100

;; Query time: 1 msec
;; SERVER: 192.168.65.1#53(192.168.65.1)
;; WHEN: Thu Feb 04 22:44:09 UTC 2021
;; MSG SIZE  rcvd: 48
```

# Prerequisite - CoreDNS

# Prerequisite - Network Namespaces

# Prerequisite - Docker Networking

# Prerequisite - CNI

# Cluster Networking

# Pod Networking

# CNI in kubernetes

# CNI weave

# IP Address Management

# Service Networking

# DNS in kubernetes

# CoreDNS in Kubernetes

# Ingress








