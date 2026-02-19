# Rust Design Patterns

## Creational Patterns

### Builder Pattern
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

### Factory Pattern
Use associated functions (static methods) as factories.

```rs
impl Widget {
    pub fn new_button() -> Self { /* ... */ }
    pub fn new_label() -> Self { /* ... */ }
}
```

## Structural Patterns

### Newtype Pattern
Wrap existing types to add type safety.

```rs
struct UserId(u64);
struct ProductId(u64);
```

### Type State Pattern
Use types to represent states.

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

## Behavioral Patterns

### Strategy Pattern
Use trait objects or generics for different strategies.

```rs
trait CompressionStrategy {
    fn compress(&self, data: &[u8]) -> Vec<u8>;
}

struct Gzip;
impl CompressionStrategy for Gzip { /* ... */ }

struct Zlib;
impl CompressionStrategy for Zlib { /* ... */ }
```

### Visitor Pattern
Use traits to separate algorithms from data structures.

### RAII Pattern
Resource Acquisition Is Initialization - automatic cleanup.

```rs
struct File {
    handle: FileHandle,
}

impl Drop for File {
    fn drop(&mut self) {
        // Cleanup happens automatically
        self.handle.close();
    }
}
```
