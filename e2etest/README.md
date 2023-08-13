- [Abstract](#abstract)
- [Materials](#materials)
- [E2E Test](#e2e-test)
- [Selenium](#selenium)
- [Cypress](#cypress)
- [Playwright](#playwright)

---

# Abstract

E2E Test (end to end test) 에 대해 적는다. 

어떤 tool 을 선택하느냐도 중요하다. 그러나 더욱 중요한 것은 테스트 시나리오가
준비되어야 한다. 보통 QA Engineer 가 요구사항을 분석하여 Test Sheet 를 제작하고
그것을 기반으로 개발자가 E2E Test 를 구현한다.

# Materials

* [뭣? 딸깍 몇 번에 웹 E2E 테스트 코드를 생성하고 수행한다고? 에러도 잡아준다고? 영상도 뽑아준다고? | naver](https://d2.naver.com/helloworld/4003712)
* [Selenium에서 Cypress로 갈아탄 후기](https://blog.hbsmith.io/selenium%EC%97%90%EC%84%9C-cypress%EB%A1%9C-%EA%B0%88%EC%95%84%ED%83%84-%ED%9B%84%EA%B8%B0-324f224c14db)

# E2E Test

* [E2E Test 알아보기](https://blog.hbsmith.io/e2e-test-%EC%95%8C%EC%95%84%EB%B3%B4%EA%B8%B0-3c524862469d)

----

사용자 입장에서 진행하는 테스트를 말한다. 보통 Web, App 등과 같은 Client 를
이용하여 모든 Endpoint 를 경유한 기능 테스트를 수행한다.

예를 들어 Frontend 의 경우 다음과 같은 것들을 테스트 한다.

* 특정 페이지를 로딩했을 때 원하는 Text 가 나오는가?
* A 를 click 하였을 때 기대하는 동작을 하는가?

Backend 의 예는 다음과 같다.

* Redis 를 Docker Container 로 띄우고 Business Logic 을 점검한다.
* Kafka 를 Docker Container 로 띄우고 Business Logic 을 점검한다.

QA 가 매번 진행하는 Regression Test 를 E2E Test 로 자동화할 수 있다.

# Selenium

모든 Browser 를 대상으로 테스트할 수 있다. 상당히 시간이 오래걸린다.
보통 릴리즈전 최종 테스트에 적당하다.

# Cypress

하나의 Browser (Chrome) 를 대상으로 테스트할 수 있다. 빠르게 테스트할 수 있다. 보통
개발 도중 테스트에 적당하다. 유료이다.

# Playwright

WIP...
