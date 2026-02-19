# Rust Style Guide

## Rust Naming Conventions

* **Modules**: Use snake_case (e.g., `my_module`)
* **Types**: Use UpperCamelCase (e.g., `MyStruct`)
* **Functions**: Use snake_case (e.g., `my_function`)
* **Constants**: Use SCREAMING_SNAKE_CASE (e.g., `MY_CONSTANT`)
* **Lifetimes**: Use short lowercase names (e.g., `'a`, `'b`)

## Code Organization

* Keep modules focused and cohesive
* Use `mod.rs` or module files for organization
* Export public APIs carefully with `pub`
* Document public APIs with `///` doc comments

## Best Practices

* Prefer immutability by default
* Use `Result` and `Option` instead of panicking
* Handle errors explicitly with `?` operator
* Use iterators and functional patterns where appropriate
* Avoid unnecessary clones
