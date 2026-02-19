# Rust Architecture

## Project Structure

```
my_project/
├── Cargo.toml          # Package manifest
├── Cargo.lock          # Dependency lock file
├── src/
│   ├── main.rs         # Binary entry point
│   ├── lib.rs          # Library entry point
│   ├── module1.rs      # Module file
│   └── module2/        # Module directory
│       ├── mod.rs      # Module declaration
│       └── submod.rs   # Submodule
├── tests/              # Integration tests
│   └── integration_test.rs
├── benches/            # Benchmarks
│   └── benchmark.rs
└── examples/           # Example binaries
    └── example.rs
```

## Layered Architecture

### Domain Layer
Core business logic with no external dependencies.

```rs
// domain/user.rs
pub struct User {
    id: UserId,
    email: Email,
}
```

### Application Layer
Use cases and application services.

```rs
// application/user_service.rs
pub struct UserService {
    repo: Box<dyn UserRepository>,
}
```

### Infrastructure Layer
External dependencies (database, HTTP, etc.).

```rs
// infrastructure/postgres_user_repo.rs
pub struct PostgresUserRepository {
    pool: PgPool,
}
```

## Hexagonal Architecture (Ports & Adapters)

Define traits (ports) for external dependencies and implement adapters.

```rs
// Ports
trait UserRepository {
    fn find(&self, id: UserId) -> Result<User>;
    fn save(&self, user: &User) -> Result<()>;
}

// Adapters
struct PostgresUserRepository;
impl UserRepository for PostgresUserRepository { /* ... */ }

struct InMemoryUserRepository;
impl UserRepository for InMemoryUserRepository { /* ... */ }
```

## Async Architecture

Use Tokio or async-std for asynchronous applications.

```rs
#[tokio::main]
async fn main() {
    let server = Server::new();
    server.run().await;
}
```

## Microservices

* Use **tonic** for gRPC services
* Use **actix-web** or **axum** for REST APIs
* Use **rdkafka** for Kafka integration
* Use **redis** for caching
