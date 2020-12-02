- [Abstract](#abstract)
- [Materials](#materials)
- [Basic Usage](#basic-usage)
  - [Useful Commands](#useful-commands)
  - [Django, Postgres](#django-postgres)
- [Advanced Usage](#advanced-usage)
  - [After updating docker-compose.yml](#after-updating-docker-composeyml)
  - [After updating Dockerfile-dev](#after-updating-dockerfile-dev)
  - [Want to delete database with volumes](#want-to-delete-database-with-volumes)

----

# Abstract

- docker container 여러 개를 한번에 실행하는 애플리케이션이다. 이런
  것을 docker orchestration 이라고한다.
- graylog 와 같이 mongodb, elasticsearch 등등 많은 container 가 필요한
  것들은 docker-compose.yml 을 구해서 테스트 해보는 것이 편하다.

# Materials

- [도커 컴포즈를 활용하여 완벽한 개발 환경 구성하기, 컨테이너 시대의 Django 개발환경 구축하기](https://www.44bits.io/ko/post/almost-perfect-development-environment-with-docker-and-docker-compose)
- [Docker Compose 커맨드 사용법](https://www.daleseo.com/docker-compose/)

# Basic Usage

## Useful Commands

```console
$ vim ~/my/docker/A/docker-compose.yml
$ cd ~/my/docker/A
$ docker-compose up
$ docker-compose down
$ docker-compose top
```

## Django, Postgres

* docker-compose.yml

```yml
version: '3'

services:
  db:
    image: postgres
    volumes:
      - ./docker/data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=sampledb
      - POSTGRES_USER=sampleuser
      - POSTGRES_PASSWORD=samplesecret
      - POSTGRES_INITDB_ARGS=--encoding=UTF-8

  django:
    build:
      context: .
      dockerfile: ./compose/django/Dockerfile-dev
    environment:
      - DJANGO_DEBUG=True
      - DJANGO_DB_HOST=db
      - DJANGO_DB_PORT=5432
      - DJANGO_DB_NAME=sampledb
      - DJANGO_DB_USERNAME=sampleuser
      - DJANGO_DB_PASSWORD=samplesecret
      - DJANGO_SECRET_KEY=dev_secret_key
    ports:
      - "8000:8000"
    command: 
      - python manage.py runserver 0:8000
    volumes:
      - ./:/app/
```

* `/compose/jango/Dockerfile-dev`

```Dockerfile
FROM python:3

RUN apt-get update && apt-get -y install \
    libpq-dev

WORKDIR /app
ADD    ./requirements.txt   /app/
RUN    pip install -r requirements.txt

# ADD    ./djangosample   /app/djangosample/
# ADD    ./manage.py      /app/

# CMD ["python", "manage.py", "runserver", "0:8000"]
```

version 3 에서 link option 이 없어도 서비스들은 하나의 네트워크에 속한다. [참고](https://docs.docker.com/compose/networking/#links)

* run

```bash
$ docker-compose up -d

# open http://127.0.0.1:8000

$ docker-compose ps

$ docker-compose stop

$ docker-compose start

# Delete services, container, network, volumes
$ docker-compose down --volume

$ docker-compose logs django
$ docker-compose logs -f django
```

* Shortcuts

```bash
alias dco='docker-compose'
alias dcb='docker-compose build'
alias dce='docker-compose exec'
alias dcps='docker-compose ps'
alias dcr='docker-compose run'
alias dcup='docker-compose up'
alias dcdn='docker-compose down'
alias dcl='docker-compose logs'
alias dclf='docker-compose logs -f'
```

# Advanced Usage

## After updating docker-compose.yml 

You need to stop, rm, up docker-compose.yml to apply new docker-compose.yml. But up is the shortcut.

```bash
$ docker-compose up -d django
```

## After updating Dockerfile-dev

Up with --build option is the shortcut.

```bash
$ docker-compose up -d --build django
```

## Want to delete database with volumes

```bash
$ docker-compose down --volume
```
