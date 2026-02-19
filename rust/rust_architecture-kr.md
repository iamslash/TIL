# Rust 아키텍처 (Architecture)

## 프로젝트 구조

```
my_project/
├── Cargo.toml          # 패키지 매니페스트
├── Cargo.lock          # 의존성 잠금 파일
├── src/
│   ├── main.rs         # 바이너리 진입점
│   ├── lib.rs          # 라이브러리 진입점
│   ├── module1.rs      # 모듈 파일
│   └── module2/        # 모듈 디렉토리
│       ├── mod.rs      # 모듈 선언
│       └── submod.rs   # 서브모듈
├── tests/              # 통합 테스트
│   └── integration_test.rs
├── benches/            # 벤치마크
│   └── benchmark.rs
└── examples/           # 예제 바이너리
    └── example.rs
```

## 계층형 아키텍처

### 도메인 레이어
외부 의존성이 없는 핵심 비즈니스 로직.

```rs
// domain/user.rs
pub struct User {
    id: UserId,
    email: Email,
}
```

### 애플리케이션 레이어
유스케이스 및 애플리케이션 서비스.

```rs
// application/user_service.rs
pub struct UserService {
    repo: Box<dyn UserRepository>,
}
```

### 인프라스트럭처 레이어
외부 의존성 (데이터베이스, HTTP 등).

```rs
// infrastructure/postgres_user_repo.rs
pub struct PostgresUserRepository {
    pool: PgPool,
}
```

## 헥사고날 아키텍처 (Ports & Adapters)

외부 의존성을 위한 trait(포트)를 정의하고 어댑터를 구현합니다.

```rs
// 포트
trait UserRepository {
    fn find(&self, id: UserId) -> Result<User>;
    fn save(&self, user: &User) -> Result<()>;
}

// 어댑터
struct PostgresUserRepository;
impl UserRepository for PostgresUserRepository { /* ... */ }

struct InMemoryUserRepository;
impl UserRepository for InMemoryUserRepository { /* ... */ }
```

## 비동기 아키텍처

비동기 애플리케이션을 위해 Tokio 또는 async-std를 사용합니다.

```rs
#[tokio::main]
async fn main() {
    let server = Server::new();
    server.run().await;
}
```

## 마이크로서비스

* **tonic** 사용 - gRPC 서비스
* **actix-web** 또는 **axum** 사용 - REST API
* **rdkafka** 사용 - Kafka 통합
* **redis** 사용 - 캐싱
