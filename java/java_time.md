- [Materials](#materials)
- [Basic](#basic)
  - [Instant](#instant)
  - [LocalTime/LocalDate/LocalDateTime](#localtimelocaldatelocaldatetime)
  - [ZoneOffset](#zoneoffset)
  - [ZoneRegion](#zoneregion)
  - [ZoneRules](#zonerules)
  - [OffsetDateTime](#offsetdatetime)
  - [ZonedDateTime](#zoneddatetime)
  - [MySQL Date and Time Type Storage Requirements](#mysql-date-and-time-type-storage-requirements)

---

# Materials

* [What's the difference between Instant and LocalDateTime? @ stackoverflow](https://stackoverflow.com/questions/32437550/whats-the-difference-between-instant-and-localdatetime)
* [(Java8) 날짜와 시간 API](https://perfectacle.github.io/2018/09/26/java8-date-time/)
* [Falsehoods programmers believe about time](https://infiniteundo.com/post/25326999628/falsehoods-programmers-believe-about-time)
  * 시간에 관한 거짓들

# Basic

## Instant

moment 를 nano-seconds 로 표현한다. 

```java
// 2018-09-26T11:41:56.281466
final var localDateTimeNow = LocalDateTime.now();
// Unix Timestamp is based on UTC. We can get Unix Timestamp from LocalDateTime with Time Zone.
// Because LocalDateTime has no Zone.
// Unix Timestamp is UTC(+00:00). UTC+09:00 means KST(Asia/Seoul) - 9 hours.
// 2018-09-26T02:41:56.281466Z
final var instantFromAsiaSeoulLocalDateTime = localDateTimeNow.atZone(ZoneId.of("Asia/Seoul")).toInstant();
// 2018-09-26T02:41:56.281466Z
final var instantFromAsiaSeoulLocalDateTime2 = Instant.from(localDateTimeNow.atZone(ZoneId.of("Asia/Seoul")));

// Unix Timestamp is UTC(+00:00). UTC(+00:00) has same time in GMT.
// 2018-09-26T11:41:56.281466Z
final var instantFromGMTLocalDateTime = localDateTimeNow.atZone(ZoneId.of("GMT")).toInstant();
// 2018-09-26T11:41:56.281466Z
final var instantFromGMTLocalDateTime2 = Instant.from(localDateTimeNow.atZone(ZoneId.of("GMT")));

// System Default Time is KST(UTC+09:00).
// 2018-09-26T02:41:56.281834Z
final var instantFromZonedDateTime = ZonedDateTime.now().toInstant();
// 2018-09-26T02:41:56.281933Z
final var instantFromAsiaSeoulZonedDateTime = ZonedDateTime.now(ZoneId.of("Asia/Seoul")).toInstant();
// 2018-09-26T02:41:56.281884Z
final var instantFromGMTZonedDateTime = ZonedDateTime.now(ZoneId.of("GMT")).toInstant();

// System Default Time is KST(UTC+09:00).
// 2018-09-26T02:41:56.281834Z
final var instantFromOffsetDateTime = OffsetDateTime.now().toInstant();
// 2018-09-26T02:41:56.281834Z
final var instantFromUTC9OffsetDateTime = OffsetDateTime.now(ZoneOffset.of("+9")).toInstant();
// 2018-09-26T02:41:56.281834Z
final var instantFromUTCOffsetDateTime = OffsetDateTime.now(ZoneOffset.UTC).toInstant();

// Easy way to get Unix Timestamp.
// 2018-09-26T02:41:56.281834Z
final var instantNow = Instant.now();

// System Default Time Offset is UTC+09:00.
// true, They are all same except LocalDateTime UTC
final var allAreSame = new HashSet<>((List.of(instantFromAsiaSeoulLocalDateTime.getEpochSecond(),
                                                instantFromAsiaSeoulLocalDateTime2.getEpochSecond(),
                                                instantFromZonedDateTime.getEpochSecond(),
                                                instantFromGMTZonedDateTime.getEpochSecond(),
                                                instantFromAsiaSeoulZonedDateTime.getEpochSecond(),
                                                instantFromOffsetDateTime.getEpochSecond(),
                                                instantFromUTCOffsetDateTime.getEpochSecond(),
                                                instantFromUTC9OffsetDateTime.getEpochSecond(),
                                                instantNow.getEpochSecond()))).size() == 1;

```

## LocalTime/LocalDate/LocalDateTime

Zone Offset, Zone Region 이 없는 Time API 이다. 

```java
// 1993-05-30T01:05:30
final var birthDateTime = LocalDateTime.of(1993, 5, 30, 1, 5, 30);
final var birthDate = LocalDate.of(1993, 5, 30);
final var birthTime = LocalTime.of(1, 5, 30);
final var birthDay = LocalDateTime.of(birthDate, birthTime);
```

## ZoneOffset

UTC 기준으로 Time Offset 을 표현하는 Time API 이다. KST 는 UTC 보다 9 시간 빠르다. 이 것을 ZoneOffset 으로 표기하면 `UTC +09:00` 와 같다. ZoneOffset 은 ZoneId 의 자식 Class 이다.

```java
// UTC +09:00
final var zoneOffset = ZoneOffset.of("+9");
final var zoneOffset2 = ZoneOffset.of("+09");
final var zoneOffsetIso8601Format = ZoneOffset.of("+09:00");
final var zoneOffset3 = ZoneOffset.of("+09:00:00");
final var zoneOffset4 = ZoneId.of("+9");
final var zoneOffset5 = ZoneId.of("+09");
final var zoneOffsetIso8601Format2 = ZoneId.of("+09:00");
final var zoneOffset6 = ZoneId.of("+09:00:00");

// UTC ±00:00
final var zoneOffset7 = ZoneOffset.of("+0");
final var zoneOffset8 = ZoneOffset.of("-0");
final var zoneOffset9 = ZoneOffset.of("+00");
final var zoneOffset10 = ZoneOffset.of("-00");
final var zoneOffsetIso8601Format3 = ZoneOffset.of("+00:00");
final var zoneOffsetIso8601Format4 = ZoneOffset.of("-00:00");
final var zoneOffsetIso8601Format5 = ZoneOffset.of("Z"); // Zulu Time
final var zoneOffset11 = ZoneOffset.of("+00:00:00");
final var zoneOffset12 = ZoneOffset.of("-00:00:00");
final var zoneOffset13 = ZoneId.of("+0");
final var zoneOffset14 = ZoneId.of("-0");
final var zoneOffset15 = ZoneId.of("+00");
final var zoneOffset16 = ZoneId.of("-00");
final var zoneOffsetIso8601Format6 = ZoneId.of("+00:00");
final var zoneOffsetIso8601Format7 = ZoneId.of("-00:00");
final var zoneOffsetIso8601Format8 = ZoneId.of("Z"); // Zulu Time
final var zoneOffset17 = ZoneId.of("+00:00:00");
final var zoneOffset18 = ZoneId.of("-00:00:00");
```

## ZoneRegion

Time Zone 을 의미한다. KST 의 ZoneRegion 은 `Asia/Seoul` 으로 표기한다. ZoneRegion 은 ZoneId 의 자식 Class 이다.

```java
// KST
final var zoneId = ZoneId.of("Asia/Seoul");
final var zoneId2 = ZoneId.of("UTC+9");
final var zoneId3 = ZoneId.of("UTC+09");
final var zoneId4 = ZoneId.of("UTC+09:00");
final var zoneId5 = ZoneId.of("UTC+09:00:00");
final var zoneId6 = ZoneId.of("GMT+9");
final var zoneId7 = ZoneId.of("GMT+09");
final var zoneId8 = ZoneId.of("GMT+09:00");
final var zoneId9 = ZoneId.of("GMT+09:00:00");
final var zoneId10 = ZoneId.of("UT+9");
final var zoneId11 = ZoneId.of("UT+09");
final var zoneId12 = ZoneId.of("UT+09:00");
final var zoneId13 = ZoneId.of("UT+09:00:00");
```

## ZoneRules

DST (Daylight Saving Time, Summer Time) 과 같은 Time Transition Rule 을 말한다. 

```java
// ZoneOffset has no Time Transition Rule
ZoneOffset.of("+1").getRules().getTransitionRules().forEach(System.out::println);

// ZoneRegion has no Time Transition Rule
ZoneId.of("Africa/Brazzaville").getRules().getTransitionRules().forEach(System.out::println);

// ZoneRegion has Time Transition Rule
// Output:
// TransitionRule[Gap +01:00 to +02:00, SUNDAY on or after MARCH 25 at 02:00 STANDARD, standard offset +01:00]
// TransitionRule[Overlap +02:00 to +01:00, SUNDAY on or after OCTOBER 25 at 02:00 STANDARD, standard offset +01:00]
ZoneId.of("CET").getRules().getTransitionRules().forEach(System.out::println);
```

## OffsetDateTime

LocalDateTime 와 ZoneOffset 을 합친 Time API 이다. 축구경기중계에 사용한다.

```java
final var barca = OffsetDateTime.of(LocalDateTime.of(2018, 5, 6, 20, 45, 0), ZoneOffset.of("+2"));
// 2018-05-06T20:45+02:00
System.out.println(barca);

final var seoul = OffsetDateTime.of(LocalDateTime.of(2018, 5, 7, 3, 45, 0), ZoneOffset.of("+9"));
// 2018-05-07T03:45+09:00
System.out.println(seoul);

// There are same times
// 2018-05-06T18:45Z
System.out.println(barca.atZoneSameInstant(ZoneId.of("Z")));
// 2018-05-06T18:45Z
System.out.println(seoul.atZoneSameInstant(ZoneId.of("Z")));

// 1970-01-01T00:00Z
final var unixTimeOfUTC = OffsetDateTime.of(1970, 1, 1, 0, 0, 0, 0, ZoneOffset.UTC);
// 1970-01-01T00:00+09:00
final var unixTimeOfUTC9 = OffsetDateTime.of(1970, 1, 1, 0, 0, 0, 0, ZoneOffset.of("+9"));
// false, They have different ZoneOffset.
System.out.println(unixTimeOfUTC.equals(unixTimeOfUTC9));

// 1970-01-01T00:00
final var unixTimeOfUTCLocalDateTime = unixTimeOfUTC.toLocalDateTime();
// 1970-01-01T00:00
final var unixTimeOfUTCL9ocalDateTime = unixTimeOfUTC9.toLocalDateTime();
// true, They are same because LocalDateTime has no ZoneOffset.
System.out.println(unixTimeOfUTCLocalDateTime.equals(unixTimeOfUTCL9ocalDateTime));
```

## ZonedDateTime

OffsetDateTime 와 ZoneRegion 을 합친 Time API 이다.

```java
// 2018-03-25T01:59:59+01:00[CET]
System.out.println(ZonedDateTime.of(LocalDateTime.of(2018, 3, 25, 1, 59, 59), ZoneId.of("CET")));
// 2018-03-25T03:00+02:00[CET]
System.out.println(ZonedDateTime.of(LocalDateTime.of(2018, 3, 25, 2, 0, 0), ZoneId.of("CET")));
// 2018-10-28T02:59:59+02:00[CET]
System.out.println(ZonedDateTime.of(LocalDateTime.of(2018, 10, 28, 2, 59, 59), ZoneId.of("CET")));
// 2018-10-28T03:00+01:00[CET]
System.out.println(ZonedDateTime.of(LocalDateTime.of(2018, 10, 28, 3, 0, 0), ZoneId.of("CET")));

// These are same because ZoneRegion, ZoneOffset has no Time Transition Rule
// 2018-06-01T00:00+09:00[Asia/Seoul]
System.out.println(ZonedDateTime.of(LocalDateTime.of(2018, 6, 1, 0, 0, 0), ZoneId.of("Asia/Seoul")));
// 2018-12-01T00:00+09:00[Asia/Seoul]
System.out.println(ZonedDateTime.of(LocalDateTime.of(2018, 12, 1, 0, 0, 0), ZoneId.of("Asia/Seoul")));
// 2018-06-01T00:00+09:00
System.out.println(ZonedDateTime.of(LocalDateTime.of(2018, 6, 1, 0, 0, 0), ZoneId.of("+9")));
// 2018-12-01T00:00+09:00
System.out.println(ZonedDateTime.of(LocalDateTime.of(2018, 12, 1, 0, 0, 0), ZoneId.of("+9")));

final var zonedDateTimeOfSeoul = ZonedDateTime.of(2018, 1, 1, 0, 0, 0, 0, ZoneId.of("Asia/Seoul"));
final var zonedDateTimeOfTokyo = ZonedDateTime.of(2018, 1, 1, 0, 0, 0, 0, ZoneId.of("Asia/Tokyo"));
// false, They have different TimeZone.
System.out.println(zonedDateTimeOfSeoul.equals(zonedDateTimeOfTokyo));

final var offsetDateTimeOfSeoul = zonedDateTimeOfSeoul.toOffsetDateTime();
final var offsetDateTimeOfTokyo = zonedDateTimeOfTokyo.toOffsetDateTime();
// true, They have same Offset, different Region but OffsetDateTime has no ZoneRegion but ZoneOffset.
System.out.println(offsetDateTimeOfSeoul.equals(offsetDateTimeOfTokyo));

final var zonedDateTimeOfWinter = ZonedDateTime.of(2018, 1, 1, 0, 0, 0, 0, ZoneId.of("CET"));
final var zonedDateTimeOfSummer = ZonedDateTime.of(2018, 6, 1, 0, 0, 0, 0, ZoneId.of("CET"));
// true, They have same ZoneRegion such as CET.
System.out.println(zonedDateTimeOfWinter.getZone().equals(zonedDateTimeOfSummer.getZone()));
// false, Offset is +01:00 in winter, +02:00 in summer.
System.out.println(zonedDateTimeOfWinter.getOffset().equals(zonedDateTimeOfSummer.getOffset()));

final var offsetDateTimeOfWinter = zonedDateTimeOfWinter.toOffsetDateTime();
final var offsetDateTimeOfSummer = zonedDateTimeOfSummer.toOffsetDateTime();
// false, They have no ZoneRegion and different Offset. 
System.out.println(offsetDateTimeOfWinter.getOffset().equals(offsetDateTimeOfSummer.getOffset()));
```

## MySQL Date and Time Type Storage Requirements

* [11.7 Data Type Storage Requirements | mysql](https://dev.mysql.com/doc/refman/8.0/en/storage-requirements.html)
* [Java LocalDateTime과 MySQL DATETIME](https://sungjk.github.io/2022/08/13/localdatetime-datetime.html)

----

MySQL 의 Time 관련 Data Type 들의 크기는 다음과 같다.

| Data Type | Storage Required Before MySQL 5.6.4 |	Storage Required as of MySQL 5.6.4 |
|--|--|--|
| YEAR	| 1 byte	| 1 byte |
| DATE	| 3 bytes	| 3 bytes|
| TIME	| 3 bytes	| 3 bytes + fractional seconds storage |
| DATETIME	| 8 bytes	| 5 bytes + fractional seconds storage |
| TIMESTAMP	| 4 bytes	| 4 bytes + fractional seconds storage |

fractional seconds 는 초 아래 몇자리 까지 표현가능한지를 의미한다. 다음은 fractional seconds 종류에 따른 data size 이다.

| Fractional Seconds Precision |	Storage Required |
|--|--|
|0 |	0 bytes |
|1, 2 |	1 byte  |
|3, 4 |	2 bytes |
|5, 6 |	3 bytes |

`datetime(6)` 는 초 아래 6 자리 까지 표현한다라는 의미이다. 즉, micro-seconds 까지 표현가능하다라는 의미이다. data-size 는 6 bytes 이다.

Java 의 Instant 는 초 아래 9 자리 까지 표현가능하다. 즉, nano-seconds 까지 표현가능하다. MySQL 에 `datetime(6)` 으로 저장하자.
