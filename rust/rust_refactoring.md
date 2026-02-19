# Rust Refactoring

## Common Refactoring Patterns

### Extract Method
Break down large functions into smaller, focused ones.

### Replace Clone with References
Use borrowing instead of cloning when possible.

### Introduce Type Parameter
Replace concrete types with generics for reusability.

### Extract Trait
Create traits to abstract common behavior.

### Replace Error Handling
Migrate from `unwrap()` to proper error propagation with `?`.

## Refactoring Tools

* **rustfmt**: Automatic code formatting
* **clippy**: Linting and suggestions
* **rust-analyzer**: IDE support with refactoring actions
