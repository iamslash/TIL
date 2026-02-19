# Rust 디자인 패턴 (Design Patterns)

## 생성 패턴

### 빌더 패턴
```rs
pub struct Config {
    host: String,
    port: u16,
    timeout: u64,
}

impl Config {
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::default()
    }
}

pub struct ConfigBuilder {
    host: String,
    port: u16,
    timeout: u64,
}

impl ConfigBuilder {
    pub fn host(mut self, host: String) -> Self {
        self.host = host;
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn build(self) -> Config {
        Config {
            host: self.host,
            port: self.port,
            timeout: self.timeout,
        }
    }
}
```

### 팩토리 패턴
연관 함수(정적 메서드)를 팩토리로 사용합니다.

```rs
impl Widget {
    pub fn new_button() -> Self { /* ... */ }
    pub fn new_label() -> Self { /* ... */ }
}
```

## 구조 패턴

### 뉴타입 패턴
기존 타입을 래핑하여 타입 안전성을 추가합니다.

```rs
struct UserId(u64);
struct ProductId(u64);
```

### 타입 상태 패턴
타입을 사용하여 상태를 나타냅니다.

```rs
struct Locked;
struct Unlocked;

struct Door<State> {
    state: PhantomData<State>,
}

impl Door<Locked> {
    fn unlock(self) -> Door<Unlocked> { /* ... */ }
}

impl Door<Unlocked> {
    fn lock(self) -> Door<Locked> { /* ... */ }
}
```

## 행동 패턴

### 전략 패턴
다양한 전략을 위해 trait 객체 또는 제네릭을 사용합니다.

```rs
trait CompressionStrategy {
    fn compress(&self, data: &[u8]) -> Vec<u8>;
}

struct Gzip;
impl CompressionStrategy for Gzip { /* ... */ }

struct Zlib;
impl CompressionStrategy for Zlib { /* ... */ }
```

### 방문자 패턴
데이터 구조에서 알고리즘을 분리하기 위해 trait를 사용합니다.

### RAII 패턴
리소스 획득은 초기화 - 자동 정리.

```rs
struct File {
    handle: FileHandle,
}

impl Drop for File {
    fn drop(&mut self) {
        // 자동으로 정리 발생
        self.handle.close();
    }
}
```
