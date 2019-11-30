# Abstract

logging agent 인 collectd 와 influxdb, grafana 를 연동하여 centralized logging system 을 구성해본다.

# Materials

* [How To Configure Collectd to Gather System Metrics for Graphite on Ubuntu 14.04 @ digitalocean](https://www.digitalocean.com/community/tutorials/how-to-configure-collectd-to-gather-system-metrics-for-graphite-on-ubuntu-14-04)
* [collectd @ github](https://github.com/collectd/collectd)

# Install

## Install collectd-influxdb-grafana on docker 

* [Try InfluxDB and Grafana by docker](https://blog.laputa.io/try-influxdb-and-grafana-by-docker-6b4d50c6a446)
  * [src](https://github.com/justlaputa/collectd-influxdb-grafana-docker)

----

```bash
$ cd ~/my/docker
$ git clone git@github.com:justlaputa/collectd-influxdb-grafana-docker.git
$ cd collectd-influxdb-grafana-docker.git
$ docker-compose up -d
# http://localhost:8083 influxdb admin page
# http://localhost:3000 grafana web page (login with admin/admin)
```

* It doesn't work on Win7 So I installed [Collectd Win](https://ssc-serv.com/download.shtml) on Win 7. And It worked. But Free Version can send a data just every 5 min.

# Config for Collectd

* [collectd로 system resource 모니터링](https://kbss27.github.io/2017/05/04/collectd/)

-----

* `/etc/collectd.conf`

```conf
--- Loadplugin section ---

LoadPlugin cpu
LoadPlugin df
LoadPlugin disk
LoadPlugin interface
LoadPlugin load
LoadPlugin memory
LoadPlugin network

--- Plugin configuration ---
<Plugin cpu>
  ReportByCpu true
  ReportByState true
  ValuesPercentage true
</Plugin>

...

<Plugin network>
#       # client setup:
        Server "127.0.0.1" "25826"
#       <Server "127.0.0.1" "25826">
#               SecurityLevel Encrypt
#               Username "user"
#               Password "secret"
#               Interface "eth0"
#               ResolveInterval 14400
#       </Server>
#       TimeToLive 128
#
#       # server setup:
#       Listen "127.0.0.1" "25826"
#       <Listen "239.192.74.66" "25826">
#               SecurityLevel Sign
#               AuthFile "/etc/collectd/passwd"
#               Interface "eth0"
#       </Listen>
        MaxPacketSize 1452
#
#       # proxy setup (client and server as above):
#       Forward true
#
#       # statistics about the network plugin itself
#       ReportStats false
#
#       # "garbage collection"
#       CacheFlush 1800
```

# Config for influxdb

* [collectd로 system resource 모니터링](https://kbss27.github.io/2017/05/04/collectd/)

-----

* ` /etc/influxdb/influxdb.conf`

```conf
###
### [collectd]
###
### Controls the listener for collectd data.
###

[collectd]
  enabled = true
  bind-address = ":25826"
  database = "collectd"
  typesdb = "/usr/share/collectd/types.db"

  # These next lines control how batching works. You should have this enabled
  # otherwise you could get dropped metrics or poor performance. Batching
  # will buffer points in memory if you have many coming in.

  # batch-size = 1000 # will flush if this many points get buffered
  # batch-pending = 5 # number of batches that may be pending in memory
  # batch-timeout = "1s" # will flush at least this often even if we haven't hit buffer limit
  # read-buffer = 0 # UDP Read buffer size, 0 means OS default. UDP listener will fail if set above OS max.
```

* check the data

  * browse `http://localhost:8083`
  * `SHOW DATABASES`
  * `show measurements`
    ```
    cpu_value
    df_value
    disk_read
    disk_value
    disk_write
    interface_rx
    interface_tx
    memory_value
    processes_processes
    processes_threads
    ```
  * `select * from cpu_value`
  * `select * from memory_value`
  * `select * from df_value`