# Abstract

unit test, integration test, e2e test, regression test, performance test
에 대해 알아본다.

# Materials

* [Just Say No to More End-to-End Tests | Google](https://testing.googleblog.com/2015/04/just-say-no-to-more-end-to-end-tests.html)

# Basic

## Google Test Automation Conference

[GTAC](https://developers.google.com/google-test-automation-conference) 는 Google 에서 해마다 주최한 Conference 이다. 그러나 2017 이후로 활동이 정지됬다. [GTAC blog](https://testing.googleblog.com) 에 유용한 자료가 많다.

## Unit Test

해당 module 만 test 하는 것이다. 그 외 module 은 mocking 한다.

## Integration Test

특정 module 과 관계된 module 을 합쳐서 test 하는 것이다. 필요에 따라 특정 module 을 mocking 하기도 한다.

## E2E Test

User 의 행동을 기반으로 수행하는 test 이다. 예를 들어 User 가 Web Page 의 Button 을 Click 하면 Frontend, Backend 를 거쳐 다시 User 에게 보여지는 내용을 검증하는 test 이다.

## Regression Test

서비스가 출시되기 전 반복해서 하는 test 이다. QA 가 manual 하게 한다면 1, 2 시간 정도 소모한다.

## Performane Test

성능을 특정하는 test 이다. [loadtest](/loadtest/README.md) 참고.

## Recommendation

![](https://2.bp.blogspot.com/-YTzv_O4TnkA/VTgexlumP1I/AAAAAAAAAJ8/57-rnwyvP6g/s1600/image02.png)

[GTAC](https://developers.google.com/google-test-automation-conference) 는 위와 같이 test ratio 를 추천한다.
