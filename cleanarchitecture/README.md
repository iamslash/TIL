# Abstract

Clean Architecture 는 Robert C. Martin 에 의해 발명되었다.

여러 Layer 들이 Ring 모양으로 모여있다. code 의 의존성은 바깥에서 안으로 향한다. 예를 들면 Adapter Layer 의 code 는 Application Layer 의 code 를 접근할 수 있다. Application Layer 의 code 는 Adapter Layer 를 접근하지 않는다.

# Materials

* [The Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
* [DDD, Hexagonal, Onion, Clean, CQRS, … How I put it all together](https://herbertograca.com/2017/11/16/explicit-architecture-01-ddd-hexagonal-onion-clean-cqrs-how-i-put-it-all-together/)
  * Architecture 총정리

# Clean Architecture Overview

다음은 Clean Architecture 를 나타내는 그림이다.

![](https://blog.cleancoder.com/uncle-bob/images/2012-08-13-the-clean-architecture/CleanArchitecture.jpg)

의존성의 흐름이 가장 바깥 Layer 에서 안쪽 Layer 로 이어진다. 단 방향이다.

Layer 별로 관심사가 분리되어 있다.

바깥 쪽 부터 Infrastructure, Presentation, Application, Domain Layer 라고 해도 좋다.
