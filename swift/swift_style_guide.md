- [Materials](#materials)
- [Source File Basics](#source-file-basics)
  - [File Names](#file-names)
  - [File Encoding](#file-encoding)
  - [Whitespace Characters](#whitespace-characters)
  - [Special Escape Sequences](#special-escape-sequences)
  - [Invisible Characters and Modifiers](#invisible-characters-and-modifiers)
  - [String Literals](#string-literals)
- [Source File Structure](#source-file-structure)
  - [File Comments](#file-comments)
  - [Import Statements](#import-statements)
  - [Type, Variable, and Function Declarations](#type-variable-and-function-declarations)
  - [Overloaded Declarations](#overloaded-declarations)
  - [Extensions](#extensions)
- [General Formatting](#general-formatting)
  - [Column Limit](#column-limit)
  - [Braces](#braces)
  - [Semicolons](#semicolons)
  - [One Statement Per Line](#one-statement-per-line)
  - [Line-Wrapping](#line-wrapping)
    - [Function Declarations](#function-declarations)
    - [Type and Extension Declarations](#type-and-extension-declarations)
    - [Function Calls](#function-calls)
    - [Control Flow Statements](#control-flow-statements)
    - [Other Expressions](#other-expressions)
  - [Horizontal Whitespace](#horizontal-whitespace)
  - [Horizontal Alignment](#horizontal-alignment)
  - [Vertical Whitespace](#vertical-whitespace)
  - [Parentheses](#parentheses)
- [Formatting Specific Constructs](#formatting-specific-constructs)
  - [Non-Documentation Comments](#non-documentation-comments)
  - [Properties](#properties)
  - [Switch Statements](#switch-statements)
  - [Enum Cases](#enum-cases)
  - [Trailing Closures](#trailing-closures)
  - [Trailing Commas](#trailing-commas)
  - [Numeric Literals](#numeric-literals)
  - [Attributes](#attributes)
- [Naming](#naming)
  - [Apple’s API Style Guidelines](#apples-api-style-guidelines)
  - [Naming Conventions Are Not Access Control](#naming-conventions-are-not-access-control)
  - [Identifiers](#identifiers)
  - [Initializers](#initializers)
  - [Static and Class Properties](#static-and-class-properties)
  - [Global Constants](#global-constants)
  - [Delegate Methods](#delegate-methods)
- [Programming Practices](#programming-practices)
  - [Compiler Warnings](#compiler-warnings)
  - [Initializers](#initializers-1)
  - [Properties](#properties-1)
  - [Types with Shorthand Names](#types-with-shorthand-names)
  - [Optional Types](#optional-types)
  - [Error Types](#error-types)
  - [Force Unwrapping and Force Casts](#force-unwrapping-and-force-casts)
  - [Implicitly Unwrapped Optionals](#implicitly-unwrapped-optionals)
  - [Access Levels](#access-levels)
  - [Nesting and Namespacing](#nesting-and-namespacing)
  - [guards for Early Exits](#guards-for-early-exits)
  - [for-where Loops](#for-where-loops)
  - [fallthrough in switch Statements](#fallthrough-in-switch-statements)
  - [Pattern Matching](#pattern-matching)
  - [Tuple Patterns](#tuple-patterns)
  - [Numeric and String Literals](#numeric-and-string-literals)
  - [Playground Literals](#playground-literals)
  - [Trapping vs. Overflowing Arithmetic](#trapping-vs-overflowing-arithmetic)
  - [Defining New Operators](#defining-new-operators)
  - [Overloading Existing Operators](#overloading-existing-operators)
- [Documentation Comments](#documentation-comments)
  - [General Format](#general-format)
  - [Single-Sentence Summary](#single-sentence-summary)
  - [Parameter, Returns, and Throws Tags](#parameter-returns-and-throws-tags)
  - [Apple’s Markup Format](#apples-markup-format)
  - [Where to Document](#where-to-document)


----

# Materials

[Swift Style Guide | google](https://google.github.io/swift/)

# Source File Basics

## File Names

File names should be in **PascalCase**. For example, if you have a struct named `MyStruct`, the file should be named `MyStruct.swift`.

## File Encoding

Files should be encoded using **UTF-8**, as Swift source files support a wide range
of Unicode characters. This guarantees better compatibility and readability
across platforms and editors.

## Whitespace Characters

Whitespace characters should be used to make the code more readable. Use spaces
instead of tabs, and indent with 2 spaces per level. This ensures consistent
formatting across environments.

## Special Escape Sequences

Swift provides special escape sequences for certain characters in strings, like
`\n` for a newline character, `\t` for a tab, and `\"` for a quote character. Use
these sequences, rather than inserting the actual characters, to make the code
more readable.

```swift
let sampleString = "This is a string with a newline character: \nAnd some \"quotes\""
```

## Invisible Characters and Modifiers

Avoid using invisible characters or modifiers (e.g., zero-width spaces) in your
code, as this can lead to unexpected behavior and harder-to-read code.

## String Literals

To represent strings, use string literals, which are enclosed in double quotes
(`" "`). Multiline strings can be represented using triple double quotes (`"""
"""`).

```swift
let singleLineString = "This is a single line string."
let multilineString = """
  This is a
  multiline string.
"""
```

# Source File Structure

## File Comments

Use file comments to provide documentation on the contents of the file. Add
relevant information like `author`, `copyright`, and `license` details at the top of
each source file.

```swift
//
//  MyStruct.swift
//  MyLibrary
//
//  Created by David Sun on 07/16/2023.
//  Copyright © 2020 iamslash. All rights reserved.
//
```

## Import Statements

Place import statements at the top of the file, below the file comments. Use one
import statement per line and sort them alphabetically.

```swift
import Foundation
import UIKit
```

## Type, Variable, and Function Declarations

Begin each type of declaration (e.g., `class`, `struct`, `protocol`, and `enum`)
with a one-line description of its purpose. Include more detailed descriptions
if necessary. Use the same approach for variables and functions.

```swift
// A simple example struct
struct MyStruct {
  // A stored property
  var myProperty: Int

  // A simple example function
  func doSomething() {
    print("Hello, World!")
  }
}
```

## Overloaded Declarations

Group overloaded declarations together, and document the differences between the
different versions.

```swift
// Adds two integers and returns the result
func add(_ a: Int, _ b: Int) -> Int {
  return a + b
}

// Adds two doubles and returns the result
func add(_ a: Double, _ b: Double) -> Double {
  return a + b
}
```

## Extensions

Use extensions to separate functionality into logical chunks and to provide a
clear organization of your code. Always add a one-line description of the
purpose of the extension.

```swift
// MARK: - UI-related functionality
extension MyStruct {
  func setupUI() {
    // Setup user interface
  }
}
```

# General Formatting

## Column Limit

The column limit for Swift code is 100 characters per line, to improve readability and avoid long horizontal lines.

```swift
// Bad
let reallyLongLine = "This is a really long string that is definitely going to pass the 100 characters limit, which is not recommended."

// Good
let longLine = "This is a long string that is still within the 100 characters limit, making it easier to read."
```

## Braces

Braces should use Egyptian style, with the opening brace on the same line as the
statement or declaration and not on a new line. Same for `else` and `catch`
blocks.

```swift
// Bad
func doSomething()
{
    if true
    {
        someFunction()
    }
    else
    {
        anotherFunction()
    }
}

// Good
func doSomething() {
    if true {
        someFunction()
    } else {
        anotherFunction()
    }
}
```

## Semicolons

Semicolons should be avoided, except in the rare cases where they are necessary
(e.g., multiple statements on a single line when needed).

```swift
// Bad
let a: Int = 1;
let b: Int = 2;

// Good
let a: Int = 1
let b: Int = 2
```

## One Statement Per Line

As a general rule, each statement should be on its own line.

```swift
// Bad
let a = 1; let b = 2

// Good
let a = 1
let b = 2
```

## Line-Wrapping

### Function Declarations

Split after the `->` for return type and align with the first parameter.

```swift
func veryLongFunctionName(parameterOne: Int, parameterTwo: String, parameterThree: Double)
    -> Bool {
    // Function body
}
```

### Type and Extension Declarations

Split after the colon that separates the definition from the type.

```swift
class MyClass: AnotherReallyLongClassWithAVeryLongName,
          AnotherProtocolWithAVeryLongName {
    // Class body
}
```

### Function Calls

Break after the open parenthesis, and for each successive argument.Label the
arguments if possible.

```swift
someFunctionWithManyParameters(
    parameterOne: 1,
    parameterTwo: "string",
    parameterThree: 3.14,
    parameterFour: true
)
```

### Control Flow Statements

For `if`, `guard`, `while`, and `switch` statements, place conditions on a
single line, and break after each condition.

```swift
if conditionOne,
   conditionTwo {
    // Body
}
```

### Other Expressions

Break before operators and align them.

```swift
let result = x * (y + z)
             / w
```

## Horizontal Whitespace

Avoid unnecessary horizontal whitespace, including at the end of lines and along with operators.

```swift
// Bad
let a = 1 + 2

// Good
let a = 1+2
```

## Horizontal Alignment

Avoid using horizontal alignment, as it may make the code harder to read.

```swift
// Bad
let   a  = 1
let   bb = 2
let ccc  = 3

// Good
let a = 1
let bb = 2
let ccc = 3
```

## Vertical Whitespace

Vertical whitespace can help with readability, use one empty line after
declarations, and before and after new scopes.

```swift
func firstFunction() {
    // Function body
}

func secondFunction() {
    // Function body
}
```

## Parentheses

Group expressions with parentheses whenever clarity would suffer without them,
but avoid superfluous parentheses.

```swift
// Bad
let a = (1 + 2) * 3

// Good
let a = 1 + 2 * 3
```

# Formatting Specific Constructs

## Non-Documentation Comments

This refers to comments that are not meant for API documentation but to explain
the code for better understanding. Always use `//` for non-documentation comments.

```swift
// This comment explains the purpose of the code below
func getSquareRoot(of number: Double) -> Double {
    return sqrt(number)
}
```

## Properties

Properties should be written in a compact form using a single line whenever
possible. If not possible, use multi-line syntax.

```swift
var singleLineProperty: String = "Hello World"
var multiLineProperty: String = {
  let hello = "Hello",
      world = "World"
  return "\(hello) \(world)"
}()
```

## Switch Statements

Switch statements should include a newline for the cases and their associated
code.

```swift
switch value {
case 1:
    print("Value is 1")
case 2:
    print("Value is 2")
default:
    print("Other value")
}
```

## Enum Cases

Enum cases should follow a lowercase naming convention and be indented from the
enum name.

```swift
enum Direction {
    case north
    case south
    case east
    case west
}
```

## Trailing Closures

Trailing closures should only be used if there is a single closure parameter.

```swift
array.map { value in
    value * 2
}
```

## Trailing Commas

Always use trailing commas in multiline arrays, dictionaries, and tuple elements
to keep diffs cleaner.

```swift
let array = [
    1,
    2,
    3,
]
```

## Numeric Literals

Numeric literals should have separators to make them visually clearer.

```swift
let largeNumber = 1_000_000
```

## Attributes

Attributes should be placed on a separate line above the declaration when possible. Multiple attributes for the same declaration should be on the same line.

```swift
@IBOutlet weak var button: UIButton!
@objc func onClick() {
}
```

# Naming

## Apple’s API Style Guidelines

Follow [Apple's API style guidelines for
Swift](https://www.swift.org/documentation/api-design-guidelines/) naming
conventions, using concise and clear 

```swift
func computeSquareRoot(of value: Double) -> Double
```

## Naming Conventions Are Not Access Control

Using an underscore as a prefix is discouraged to denote access levels. Instead,
use the appropriate access control keywords.

```swift
private var counter = 0
```

## Identifiers

Identifiers names should be clear and descriptive.

```swift
var employeeList: [Employee]
func getDiscountPrice(for item: Item) -> Double

```

## Initializers

Initializers should generally delegate across the type and up the inheritance
chain.

```swift
class Employee {
    init(firstName: String, lastName: String) {
        self.firstName = firstName
        self.lastName = lastName
    }
}
```

## Static and Class Properties

Use "shared" as the keyword for static or class properties that return an instance.

```swift
class DateFormatter {
    static let shared = DateFormatter()
}
```

## Global Constants

Global constants should be in UPPER_CASE with an underscore as a separator.

```swift
let MAX_CONNECTIONS = 10
```

## Delegate Methods

Delegate methods should include the source object as a parameter.

```swift
func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath)
```

# Programming Practices

## Compiler Warnings

Compiler warnings provide alerts to potential issues in your code. Address all
warnings generated by the compiler to avoid potential pitfalls and ensure your
code is more stable and optimized.

```swift
let name = "John"
let age = 30 // Warning: 'age' is not used
```

## Initializers

Initializers are special methods that set up the initial state of an object or
value. Ensure that all properties have default values in Swift initializers.

```swift
class Apple {
    var color: String
    var weight: Double

    init(color: String, weight: Double) {
        self.color = color
        self.weight = weight
    }
}
```

## Properties

Properties associate values with a particular structure, class, or enumeration.
Use `var` for mutable properties and `let` for immutable properties.

```swift
class Person {
    let name: String
    var age: Int

    init(name: String, age: Int) {
        self.name = name
        self.age = age
    }
}
```

## Types with Shorthand Names

Use the Swift shorthand names when referring to types.

```swift
var names: [String] = ["John", "Jane"]
var ages: [Int: String] = [30: "John", 25: "Jane"]
```

## Optional Types

Optional types allow a variable to have a value or be `nil`. Use optionals when a
value might be absent.

```swift
enum Fruits {
  case apple, orange
}

let fruit: Fruits? = .apple
```

## Error Types

Error types define errors that can be thrown and caught in your program. Use
enums to create error types that conform to the `Error` protocol.

```swift
enum SerializationError: Error {
    case missing(String)
}
```

## Force Unwrapping and Force Casts

Avoid using force unwrapping and force casts, as they can cause runtime crashes.
Instead, use safe unwrapping techniques.

```swift
let value: Any = 5
if let intValue = value as? Int {
    print(intValue) 
} else {
    print("Value is not an integer.")
}
```

## Implicitly Unwrapped Optionals

**Implicitly unwrapped optionals** (**IUOs**) are used when a value is guaranteed to be
non-nil after initialization. Avoid using **IUOs** and use the non-optional type
when possible.

```swift
class Person {
    let name: String!
    let age: Int

    init(name: String?, age: Int) {
        self.name = name ?? ""
        self.age = age
    }
}
```

## Access Levels

Access levels restrict the visibility and usage of entities within your code.
Use `private`, `fileprivate`, `public`, and `open` to protect code from
unexpected manipulations.

```swift
class Pet {
    private var name: String
    fileprivate var age: Int
    public var breed: String

    init(name: String, age: Int, breed: String) {
        self.name = name
        self.age = age
        self.breed = breed
    }
}
```

## Nesting and Namespacing

Nesting is used to scope entities and prevent naming conflicts. Namespacing
organizes your code into logical areas.

```swift
struct Apple {
    struct Colors {
        static let red = "Red"
        static let green = "Green"
    }
}

let redApple = Apple.Colors.red
```

## guards for Early Exits

Use `guard` statements for **early exits** when a specific condition must be met
for the code to execute further. This ensures cleaner code by reducing the need
for nested if statements.

```swift
func greet(age: Int) {
    guard age >= 18 else {
        print("You are not old enough.")
        return
    }
    print("Welcome!")
}
```

## for-where Loops

`for-where` loops provide a concise way to filter elements in a loop.

```swift
let numbers = [1, 2, 3, 4, 5]
for number in numbers where number % 2 == 0 {
    print("\(number) is even")
}
```

## fallthrough in switch Statements

Use `fallthrough` when you want a case statement to execute the code in the next
case statement.

```swift
let value = 2
switch value {
case 1:
    print("One")
case 2:
    print("Two")
    fallthrough
case 3:
    print("Three")
default:
    print("Other")
}
```

## Pattern Matching

Pattern matching allows code to execute if certain patterns are found. Use
pattern matching to simplify complex conditions.

```swift
let shape = ("square", 4)
switch shape {
case ("square", 4):
    print("It's a square.")
default:
    print("Not a square.")
}
```

## Tuple Patterns

Tuples group multiple values into a single compound value. Tuple patterns allow
you to extract data from a tuple.

```swift
let person = (name: "John", age: 25)
switch person {
case (let name, _):
    print("Name is \(name)")
}
```

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