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








