# Abstract

Kibana 는 Elasticsearch 의 시각화 도구이다.

# Install Kibana 4.6.6 from debian package

This is matched with Elasticsearch 2.4.1. You can download from [Elastic past release](https://www.elastic.co/kr/downloads/past-releases).

```bash
$ wget https://download.elastic.co/kibana/kibana/kibana-4.6.6-amd64.deb
$ sudo dpkg -i kibana-4.6.6-amd64.deb
$ sudo vim /opt/kibana/config/kibana.yml
elasticsearch.url: "http://xxx.xxx.xxx.xxx:19200"

$ sudo /bin/systemctl daemon-reload
$ sudo /bin/systemctl enable kibana.service
  
$ sudo systemctl start kibana.service
```

# Basic

## Useful Queries

> [Kibana Query Language](https://www.elastic.co/guide/en/kibana/master/kuery-query.html)

```
# Search name is foo with date range.
> person.name:foo AND billing.dt:[now-2d/d TO now-2d/d+1h]
```
