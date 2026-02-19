# Effective Rust

## Ownership and Borrowing

1. **Prefer borrowing over ownership transfer** - when you don't need ownership
2. **Use lifetimes explicitly** - when references are involved
3. **Avoid unnecessary clones** - use references and slices
4. **Return owned values** from functions when necessary

## Error Handling

1. **Use `Result<T, E>`** - for recoverable errors
2. **Use `panic!`** - only for unrecoverable errors
3. **Implement `From` trait** - for error conversions
4. **Use `thiserror` or `anyhow` crates** - for better error handling

## Performance

1. **Prefer iterators** - they optimize better than loops
2. **Use `&str`** - instead of `String` when possible
3. **Avoid allocations in hot paths**
4. **Use `Vec::with_capacity`** - when size is known
5. **Profile before optimizing** - use tools like `cargo flamegraph`

## API Design

1. **Return `impl Trait`** - for flexibility
2. **Use builder pattern** - for complex constructors
3. **Implement standard traits** (`Debug`, `Display`, `Clone`, etc.)
4. **Use `Cow<str>`** - for flexible string APIs

## Testing

1. **Write unit tests** - with `#[test]`
2. **Use `#[cfg(test)]` modules** - for test code
3. **Write integration tests** - in `tests/` directory
4. **Use `cargo test`** - for running tests
5. **Document examples** - they also serve as tests
