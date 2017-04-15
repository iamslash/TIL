# intro

- docker container여러 개를 한번에 실행하는 애플리케이션이다. 이런
  것을 docker orchestration이라고한다.
- graylog와 같이 mongodb, elasticsearch등등 많은 container가 필요한
  것들은 docker-compose.yml을 구해서 테스트 해보는 것이 편하다.

# usage

- emacs ~/my/docker/A/docker-compose.yml
- cd ~/my/docker/A
- docker-compose up
- docker-compose down
- docker-compose top
