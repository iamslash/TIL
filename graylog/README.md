# Abstraction

- opensource log management 애플리케이션이다.
- mongodb, elasticsearch, graylog-server, graylog-web으로 구성되어 있다.
- docker-compose 를 이용하면 간단히 설치 및 실행할 수 있다. 
- mongodb는 메타데이터 저장용이다.
- elasticsearch는 로그데이터 저장용이다. full text search를 지원한다.

# Query

* [Searching](https://docs.graylog.org/en/3.1/pages/queries.html)

```bash
# Messages that include the term ssh:
> ssh

# 
> ssh login

> "ssh login"

> type:ssh

> type:(ssh OR login)

> type:"ssh login"

> _exists_:type

> NOT _exists_:type

# Messages that match regular expression
> /ethernet[0-9]+/

# By default all terms or phrases are OR connected so all messages that have at least one hit are returned. You can use Boolean operators and groups for control over this:
> "ssh login" AND source:example.org

> "ssh login" AND NOT source:example.org

# Note that AND, OR, and NOT are case sensitive and must be typed in all upper-case.

# Wildcards: Use ? to replace a single character or * to replace zero or more characters:
> source:*.org
> source:exam?le.org
> source:exam?le.*

# Note that leading wildcards are disabled to avoid excessive memory consumption! You can enable them in your Graylog configuration file:
allow_leading_wildcard_searches = true

# ???
> ssh logni~
> source:example.org~
```

# graylog install

- [docker-compose.yml](https://hub.docker.com/r/graylog2/server/)

# graylog usage

- run

  - prepare directories

```bash
mkdir ~/my/docker/graylog/config
mkdir ~/my/docker/graylog/data/mongo
mkdir ~/my/docker/graylog/data/elasticsearch
mkdir ~/my/docker/graylog/data/journal

```

  - download config files

```bash
mkdir ~/my/docker/graylog/config
cd ~/my/docker//graylog/config
wget https://raw.githubusercontent.com/Graylog2/graylog2-images/2.1/docker/config/graylog.conf
wget https://raw.githubusercontent.com/Graylog2/graylog2-images/2.1/docker/config/log4j2.xml
```
  
  - vim ~/my/docker/graylog/docker-compose.yml

```
version: '2'
services:                      
  some-mongo:                  
    image: "mongo:3"
    volumes:                   
      - /Users/iamslash/my/docker/graylog/data/mongo:/data/db
  some-elasticsearch:          
    image: "elasticsearch:2"   
    command: "elasticsearch -Des.cluster.name='graylog'"
    volumes:
      - /Users/iamslash/my/docker/graylog/data/elasticsearch:/usr/share/elasticsearch/data
  graylog:                     
    image: graylog2/server:2.1.1-1  
    volumes:
      - /Users/iamslash/my/docker/graylog/data/journal:/usr/share/graylog/data/journal
      - /Users/iamslash/my/docker/graylog/config:/usr/share/graylog/data/config
    environment:
      GRAYLOG_PASSWORD_SECRET: somepasswordpepper
      GRAYLOG_ROOT_PASSWORD_SHA2: 8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918
      GRAYLOG_WEB_ENDPOINT_URI: http://127.0.0.1:9000/api
    links:
      - some-mongo:mongo
      - some-elasticsearch:elasticsearch
    ports:
      - "9000:9000"
      - "12201/udp:12201/udp"
      - "1514/udp:1514/udp"
```
  
  - cd ~/my/docker/graylog/
  - docker-compose up

- syslog 수집하기
  - graylog의 input에 syslog UDP를 제작하자. port는 1514로 하자.
  - mac의 /etc/syslog.conf를 수정하자.
```
# Note that flat file logs are now configured in /etc/asl.conf

install.*						@127.0.0.1:32376
*.*						@127.0.0.1:1514
```
  - mac의 syslogd를 다시 실행하자.
    - sudo launchctl unload /System/Library/LaunchDaemons/com.apple.syslogd.plist
    - sudo launchctl load /System/Library/LaunchDaemons/com.apple.syslogd.plist
  - 잘 안된다. 왜지???

- GELF udp 수집하기
  - graylog의 input에 GELF UDP를 제작하자. port는 12201로 하자.
  - pip install gelfclient
  - vim ~/my/docker/graylog/client/a.py
  
```py
from gelfclient import UdpClient

gelf_server = 'localhost'

# Using mandatory arguments
gelf = UdpClient(gelf_server)

# Using all arguments
gelf = UdpClient(gelf_server, port=12201, mtu=8000, source='macbook.local')

# Bare minimum is to send a string, which will map to gelf['short_message']
gelf.log('server is DOWN')

# 'source' and 'host' are the same. Defaults to socket.gethostname() but can be overridden
gelf.log('server is DOWN', source='hostchecker')

# Set extra fields
gelf.log('status change', _state='DOWN', _server='macbook')

# Set severity level
import syslog
gelf.log('unexpected error', level=syslog.LOG_CRIT)

# You can also prepare all data into a dictionary and give that to .log
data = {}
data['short_message'] = 'warning from python'
data['host'] = 'hostchecker'
data['level'] = syslog.LOG_WARNING
gelf.log(data)
```
  - 잘되는 걸
  
# conclusion

- 
