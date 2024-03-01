
- [Abstract](#abstract)
- [Setting Up](#setting-up)
- [CDC From MySQL to Redis](#cdc-from-mysql-to-redis)

-----

# Abstract

Debezium은 변경 데이터 캡처(CDC)를 위한 오픈 소스 분산 플랫폼입니다. 데이터베이스에서 다른 애플리케이션에 의해 커밋된 모든 삽입, 업데이트, 삭제를 모니터링할 수 있게 해줍니다. Debezium을 설정함으로써, 데이터베이스의 변경 사항에 대해 실시간으로 반응할 수 있는 애플리케이션을 만들 수 있습니다. Debezium은 내구성과 속도로 알려져 있어, 심지어 실패 상황에서도 이벤트를 놓치지 않고 애플리케이션이 즉각적으로 반응할 수 있도록 보장합니다.

Debezium은 Apache Kafka 위에 구축되어 있으며, Kafka Connect 호환 커넥터를 사용하여 작업을 용이하게 합니다. 이 설계를 통해 데이터베이스 변경 사항을 이벤트 스트림으로 변환할 수 있습니다. 이 스트림들을 통해 애플리케이션은 데이터베이스의 행 수준 변경을 감지하고 즉시 반응할 수 있습니다. 다양한 데이터베이스를 지원하는 커넥터를 통해, Debezium은 다양한 환경에서 CDC를 구현하기 위한 다재다능한 도구가 됩니다.

# Setting Up

Debezium은 Apache Kafka와 Kafka Connect를 기반으로 하기 때문에, Kafka Connect의 커맨드 라인 인터페이스(CLI) 또는 구성 파일을 사용하여 Debezium 커넥터를 설정하고 관리할 수 있습니다. 일반적인 절차는 다음과 같습니다:

- Kafka 및 Kafka Connect 설치: Debezium을 실행하기 전에, Apache Kafka와 Kafka Connect가 설치되어 있어야 합니다. Kafka는 데이터 스트림을 관리하고, Kafka Connect는 데이터베이스와 Kafka 사이의 데이터 흐름을 관리합니다.
- Debezium 커넥터 구성: Debezium은 Kafka Connect를 사용하여 데이터베이스 변경 사항을 캡처합니다. 각 데이터베이스 유형(예: MySQL, PostgreSQL, MongoDB 등)에 대해 Debezium은 특정 커넥터를 제공합니다. 이 커넥터를 구성하기 위해 JSON 형식의 구성 파일을 작성합니다. 이 파일에는 데이터베이스 연결 정보, 캡처할 데이터베이스 또는 테이블, Kafka 토픽 설정 등이 포함됩니다.
- Kafka Connect에 커넥터 추가: 구성 파일을 사용하여 Kafka Connect에 Debezium 커넥터를 추가합니다. 이는 보통 curl 명령어를 사용해 Kafka Connect의 REST API를 통해 이루어집니다. 예를 들어, 커넥터 구성 파일을 Kafka Connect에 등록하는 커맨드는 다음과 같습니다:

```bash
curl -X POST -H "Content-Type: application/json" --data '@config.json' http://localhost:8083/connectors
```

여기서 config.json은 커넥터 구성 파일입니다.

- 모니터링 및 관리: 커넥터가 성공적으로 추가되고 나면, Kafka Connect와 Debezium은 지정된 데이터베이스의 변경 사항을 실시간으로 모니터링하고 Kafka 토픽으로 전송합니다. Kafka Connect의 REST API를 사용하여 커넥터의 상태를 확인하고 관리할 수 있습니다.

Debezium을 사용하면 데이터베이스 변경 사항을 효과적으로 캡처하고 다른 시스템과 동기화할 수 있으며, 이 모든 작업은 커맨드 라인을 통해 수행됩니다.

# CDC From MySQL to Redis

Debezium으로 MySQL의 hotel 데이터베이스 내 inventory_room_types 테이블의 변경 사항을 캡처하고 이를 Redis에 캐싱하기 위한 접근 방식은 몇 단계로 나뉩니다. 첫째, Debezium을 사용하여 MySQL의 변경 사항을 캡처합니다. 둘째, 이 변경 사항을 Kafka를 통해 스트림합니다. 셋째, Kafka Connect를 사용하여 이 스트림을 Redis로 전송하거나, Kafka Streams 또는 다른 스트림 처리 툴을 사용하여 처리 후 Redis에 캐싱합니다. 이 예시에서는 첫 번째 단계인 Debezium을 사용하여 MySQL 변경 사항을 캡처하기 위한 config.json 파일을 제공합니다.

```json
{
  "name": "inventory-room-types-connector",
  "config": {
    "connector.class": "io.debezium.connector.mysql.MySqlConnector",
    "tasks.max": "1",
    "database.hostname": "localhost",
    "database.port": "3306",
    "database.user": "dbuser",
    "database.password": "dbpassword",
    "database.server.id": "12345",
    "database.server.name": "hotelDbServer",
    "database.include.list": "hotel",
    "table.include.list": "hotel.inventory_room_types",
    "database.history.kafka.bootstrap.servers": "localhost:9092",
    "database.history.kafka.topic": "dbhistory.inventory_room_types",
    "transforms": "unwrap,dropPrefix",
    "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
    "transforms.dropPrefix.type": "org.apache.kafka.connect.transforms.RegexRouter",
    "transforms.dropPrefix.regex": "hotelDbServer.hotel.inventory_room_types",
    "transforms.dropPrefix.replacement": "inventory_room_types"
  }
}
```

이 설정에서 중요한 부분은 `table.include.list` 입니다. 이는 Debezium이 hotel 데이터베이스의 inventory_room_types 테이블에서 발생하는 변경 사항만 캡처하도록 지정합니다. transforms 부분은 캡처된 변경 사항을 처리하는 방법을 정의합니다. `ExtractNewRecordState` 변환은 변경 사항에서 새로운 상태만 추출하고, `RegexRouter` 변환은 토픽 이름을 변경합니다. 이는 Kafka에서 Redis로 데이터를 전송할 때 유용할 수 있습니다.

이 예시는 Debezium 설정의 시작점을 제공하지만, 실제로 Redis로 데이터를 전송하려면 Kafka Connect의 Redis 커넥터나 Kafka Streams, 또는 다른 스트림 처리 툴을 사용하여 Kafka 토픽의 데이터를 Redis로 전송하는 추가적인 설정 및 개발 작업이 필요합니다. Redis 캐싱에 대한 정확한 구현은 사용되는 도구와 특정 요구 사항에 따라 달라질 수 있습니다.
