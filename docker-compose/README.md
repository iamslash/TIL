# Abstract

- docker container 여러 개를 한번에 실행하는 애플리케이션이다. 이런
  것을 docker orchestration 이라고한다.
- graylog 와 같이 mongodb, elasticsearch 등등 많은 container 가 필요한
  것들은 docker-compose.yml 을 구해서 테스트 해보는 것이 편하다.

# Usage

```bash
$ vim ~/my/docker/A/docker-compose.yml
$ cd ~/my/docker/A
$ docker-compose up
$ docker-compose down
$ docker-compose top
```
