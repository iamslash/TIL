# Materials

* [Python Celery & RabbitMQ Tutorial](https://tests4geeks.com/blog/python-celery-rabbitmq-tutorial/)
  * [한글](https://kimdoky.github.io/tech/2019/01/23/celery-rabbitmq-tuto/)
  * [src](https://github.com/jimmykobe1171/celery-demo/tree/master)

# Basic

Celery is an asynchronous task queue. RabbitMQ is a message broker widely used with Celery.

Install RabbitMQ, Celery with docker-compose. You need open 5555 for flower. [Celery Docker Image](https://www.github.com/eea/eea.docker.celery)

* docker-compose.yml

```yml
version: "3"
  services:
    celery:
      image: eeacms/celery:4.3-1.0
      ports:
      - "5555:5555"
      environment:
        TZ: "Europe/Copenhagen"
        CELERY_BROKER_URL: "amqp://admin:admin@rabbit"
        CELERY_BACKEND_URL: "redis://redis"
        REQUIREMENTS: |
          nltk
          requests
        CONSTRAINTS: |
          nltk==3.4
          requests==2.21.0
        TASKS: |
          import requests
          import nltk
          nltk.download('punkt')

          @message_handler('nltk_queue')
          def handle_nltk(body):
            tokens = nltk.word_tokenize(body)
            print(tokens)
            return tokens

          @message_handler('req_queue')
          def handle_req(url):
            res = requests.get(url)
            status = res.status_code
            print("Got %s status code while calling %s" % (status, url))
            return status
      depends_on:
      - redis
      - rabbit

    redis:
      image: redis:5

    rabbit:
      image: rabbitmq:management-alpine
      ports:
      - "15672:15672"
      environment:
        TZ: Europe/Copenhagen
        RABBITMQ_DEFAULT_USER: admin
        RABBITMQ_DEFAULT_PASS: admin
```

Run flower.

```bash
$ docker-compose up -d

$ docker exec -it eeadockercelery_celery_1 /bin/bash

$ pip install flower

# Please be patient. It takes around 5mins.
$ flower -A tasks --port=5555
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt.zip.
[I 200410 13:56:29 command:136] Visit me at http://localhost:5555
[I 200410 13:56:29 command:141] Broker: amqp://admin:**@rabbit:5672//
[I 200410 13:56:29 command:144] Registered tasks:
    ['celery.accumulate',
     'celery.backend_cleanup',
     'celery.chain',
     'celery.chord',
     'celery.chord_unlock',
     'celery.chunks',
     'celery.group',
     'celery.map',
     'celery.starmap']
[I 200410 13:56:29 mixins:229] Connected to amqp://admin:**@rabbit:5672//
```

Open browser "localhost:5555"

Sending messages to RabbitMQ

```bash
$ docker run -it --rm --network=eeadockercelery_default --link=rabbit:rabbit python:2 bash
$ pip install eea.rabbitmq.client pika==0.10.0
$ apt-get update
$ apt-get install vim
$ vim a.py
from eea.rabbitmq.client import RabbitMQConnector

rabbit_config = {
  'rabbit_host': "rabbit",
  'rabbit_port': 5672,
  'rabbit_username': "admin",
  'rabbit_password': "admin"}

rabbit = RabbitMQConnector(**rabbit_config)
rabbit.open_connection()

rabbit.declare_queue("nltk_queue")
rabbit.send_message("nltk_queue", "Hello world !!!")

rabbit.declare_queue("req_queue")
rabbit.send_message("req_queue", "http://google.com")

rabbit.close_connection()
$ python a.py
```

Open browser "localhost:5555"
