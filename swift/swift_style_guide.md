# Materials

[Swift Style Guide | google](https://google.github.io/swift/)

# Source File Basics
## File Names
## File Encoding
## Whitespace Characters
## Special Escape Sequences
## Invisible Characters and Modifiers
## String Literals

# Source File Structure
## File Comments
## Import Statements
## Type, Variable, and Function Declarations
## Overloaded Declarations
## Extensions

# General Formatting
## Column Limit
## Braces
## Semicolons
## One Statement Per Line
## Line-Wrapping
### Function Declarations
### Type and Extension Declarations
### Function Calls
### Control Flow Statements
### Other Expressions
## Horizontal Whitespace
## Horizontal Alignment
## Vertical Whitespace
## Parentheses

# Formatting Specific Constructs
## Non-Documentation Comments
## Properties
## Switch Statements
## Enum Cases
## Trailing Closures
## Trailing Commas
## Numeric Literals
## Attributes

# Naming
## Apple’s API Style Guidelines
## Naming Conventions Are Not Access Control
## Identifiers
## Initializers
## Static and Class Properties
## Global Constants
## Delegate Methods

# Programming Practices
## Compiler Warnings
## Initializers
## Properties
## Types with Shorthand Names
## Optional Types
## Error Types
## Force Unwrapping and Force Casts
## Implicitly Unwrapped Optionals
## Access Levels
## Nesting and Namespacing
## guards for Early Exits
## for-where Loops
## fallthrough in switch Statements
## Pattern Matching
## Tuple Patterns
## Numeric and String Literals

Numeric literals represent integer and floating-point values. String literals
represent a sequence of characters.

```swift
let intLiteral = 42
let floatLiteral = 3.14
let stringLiteral = "Hello, world!"
```

## Playground Literals

Playground literals represent common objects in a playground, like colors,
images, and URLs. They are permitted in playground sources not in production
sources.

```swift
let color = #colorLiteral(red: 1, green: 0, blue: 0, alpha: 1)
let imageURL = #imageLiteral(resourceName: "icon.png")
```

## Trapping vs. Overflowing Arithmetic

Trapping arithmetic is used when the result of arithmetic produces an error.
Overflowing arithmetic produces the smallest possible value if the result is too
large.

```swift
let trapped = Int.max &+ 1 // Trapping arithmetic results in an error
let overflowing = Int.max &+ 1 // Overflowing arithmetic yields the smallest Int
```

## Defining New Operators

Use operator declarations to define custom operators for arithmetic, comparison,
or other functions.

```swift
infix operator ^^: AdditionPrecedence
func ^^(lhs: Int, rhs: Int) -> Int {
    return lhs * rhs
}
```

## Overloading Existing Operators

You can overload existing operators by providing new implementations for a
specific type or circumstance.

```swift
func +(lhs: String, rhs: Int) -> String {
    return lhs + String(rhs)
}

let result = "The number is: " + 42 // The number is: 42
```

# Documentation Comments

## General Format

In Swift, documentation comments are written using triple-slash (`///`) or
triple-backtick (`swift`) comments.

```swift
/// This is a documentation comment.
func foo() {
    ...
}
```

## Single-Sentence Summary

Start with a single-sentence summary that concisely describes the purpose of the
declared symbol. This helps developers understand the functionality at a glance.

```swift
/// Adds two integer values.
func add(_ a: Int, _ b: Int) -> Int {
    return a + b
}
```

## Parameter, Returns, and Throws Tags

Use `- Parameters:`, `- Returns:`, and `- Throws:` tags to describe function
parameters, return values, and thrown errors. These provide additional details
for users of the code.

```swift
/**
 Adds two integers and returns the result.

 - Parameters:
   - a: The first integer to add.
   - b: The second integer to add.
 - Returns: The sum of the two input integers.
*/
func add(_ a: Int, _ b: Int) -> Int {
    return a + b
}
```

## Apple’s Markup Format

Use [Apple’s Markup Format](https://developer.apple.com/library/archive/documentation/Xcode/Reference/xcode_markup_formatting_ref/) for documenting code compatible with Apple's tools like Xcode.

```swift
/**
 This class represents a simple 2D point in the coordinate system.

 - Author: iamslash
 - Version: 1.0
*/
class Point {
    var x: Double
    var y: Double

    /**
     Initializes the point with the given coordinates.

     - Parameters:
       - x: The x-coordinate of the point.
       - y: The y-coordinate of the point.
    */
    init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }
}
```

## Where to Document

Documentation comments should be provided for all public and open interfaces,
including classes, structures, enumerations, protocols, functions, properties,
and initializers. If an internal or fileprivate interface is particularly
complex, it's also helpful to add documentation for better code
comprehensibility.

```swift
/// Represents a 2D vector in the coordinate system.
public struct Vector2D {
    public let x: Double
    public let y: Double

    /// Initializes a new vector with the given `x` and `y` components.
    public init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }

    /// Adds two vectors and returns the resulting vector.
    public static func +(lhs: Vector2D, rhs: Vector2D) -> Vector2D {
        return Vector2D(x: lhs.x + rhs.x, y: lhs.y + rhs.y)
    }
}
```