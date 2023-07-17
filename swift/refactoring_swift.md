- [Materials](#materials)
- [Catalog](#catalog)
  - [Change Function Declaration](#change-function-declaration)
  - [Change Reference to Value](#change-reference-to-value)
  - [Collapse Hierarchy](#collapse-hierarchy)
  - [Combine functions into Class](#combine-functions-into-class)
  - [Combine Functions into Transform](#combine-functions-into-transform)
  - [Consolidate Conditional Expression](#consolidate-conditional-expression)
  - [Decompose Conditional](#decompose-conditional)
  - [Encapsulate Collection](#encapsulate-collection)
  - [Encapsulate Record (Replace Record with Data Class)](#encapsulate-record-replace-record-with-data-class)
  - [Encapsulate Variable (Encapsulate Field • Self-Encapsulate Field)](#encapsulate-variable-encapsulate-field--self-encapsulate-field)
  - [Extract Class](#extract-class)
  - [Extract Function](#extract-function)
  - [Extract Method](#extract-method)
  - [Extract Superclass](#extract-superclass)
  - [Extract Variable (Introduce Explaining Variable)](#extract-variable-introduce-explaining-variable)
  - [Hide Delegate](#hide-delegate)
  - [Inline Class](#inline-class)
  - [Inline Function (Inline Method)](#inline-function-inline-method)
  - [Inline Variable](#inline-variable)
  - [Inline Temp](#inline-temp)
  - [Introduce Assertion](#introduce-assertion)
  - [Introduce Parameter Object](#introduce-parameter-object)
  - [Introduce Special Case (Introduce Null Object)](#introduce-special-case-introduce-null-object)
  - [Move Field](#move-field)
  - [Move Function (Move Method)](#move-function-move-method)
  - [Move Statements into Function](#move-statements-into-function)
  - [Move Statements to Callers](#move-statements-to-callers)
  - [Parameterize Function (Parameterize Method)](#parameterize-function-parameterize-method)
  - [Preserve Whole Object](#preserve-whole-object)
  - [Pull Up Constructor Body](#pull-up-constructor-body)
  - [Pull Up Field](#pull-up-field)
  - [Pull Up Method](#pull-up-method)
  - [Push Down Field](#push-down-field)
  - [Push Down Method](#push-down-method)
  - [Remove Dead Code](#remove-dead-code)
  - [Remove Flag Argument (Replace Parameter with Explicit Methods)](#remove-flag-argument-replace-parameter-with-explicit-methods)
  - [Remove Middle Man](#remove-middle-man)
  - [Remove Setting Method](#remove-setting-method)
  - [Remove Subclass (Replace Subclass with Fields)](#remove-subclass-replace-subclass-with-fields)
  - [Rename Field](#rename-field)
  - [Rename Variable](#rename-variable)
  - [Replace Command with Function](#replace-command-with-function)
  - [Replace Conditional with Polymorphism](#replace-conditional-with-polymorphism)
  - [Replace Constructor with Factory Function (Replace Constructor with Factory Method)](#replace-constructor-with-factory-function-replace-constructor-with-factory-method)
  - [Replace Control Flag with Break (Remove Control Flag)](#replace-control-flag-with-break-remove-control-flag)
  - [Replace Derived Variable with Query](#replace-derived-variable-with-query)
  - [Replace Error Code with Exception](#replace-error-code-with-exception)
  - [Replace Exception with Precheck (Replace Exception with Test)](#replace-exception-with-precheck-replace-exception-with-test)
  - [Replace Function with Command (Replace Method with Method Object)](#replace-function-with-command-replace-method-with-method-object)
  - [Replace Inline Code with Function Call](#replace-inline-code-with-function-call)
  - [Replace Loop with Pipeline](#replace-loop-with-pipeline)
  - [Replace Magic Literal (Replace Magic Number with Symbolic Constant)](#replace-magic-literal-replace-magic-number-with-symbolic-constant)
  - [Replace Nested Conditional with Guard Clauses](#replace-nested-conditional-with-guard-clauses)
  - [Replace Parameter with Query (Replace Parameter with Method)](#replace-parameter-with-query-replace-parameter-with-method)
  - [Replace Primitive with Object (Replace Data Value with Object • Replace Type Code with Class)](#replace-primitive-with-object-replace-data-value-with-object--replace-type-code-with-class)
  - [Replace Query with Parameter](#replace-query-with-parameter)
  - [Replace Subclass with Delegate](#replace-subclass-with-delegate)
  - [Replace Superclass with Delegate (Replace Inheritance with Delegation)](#replace-superclass-with-delegate-replace-inheritance-with-delegation)
  - [Replace Temp with Query](#replace-temp-with-query)
  - [Replace Type Code with Subclasses (Extract Subclass • Replace Type Code with State/Strategy)](#replace-type-code-with-subclasses-extract-subclass--replace-type-code-with-statestrategy)
  - [Return Modified Value](#return-modified-value)
  - [Separate Query from Modifier](#separate-query-from-modifier)
  - [Slide Statements (Consolidate Duplicate Conditional Fragments)](#slide-statements-consolidate-duplicate-conditional-fragments)
  - [Split Loop](#split-loop)
  - [Split Phase](#split-phase)
  - [Split Variable (Remove Assignments to Parameters • Split Temp)](#split-variable-remove-assignments-to-parameters--split-temp)
  - [Substitute Algorithm](#substitute-algorithm)

----

# Materials

* [Refactoring | martinfowler](https://refactoring.com/catalog/)

# Catalog

## Change Function Declaration

Changing a function's name, parameters or return type to make it more
descriptive, or to follow the code style guidelines.

```swift
// bad
func sq(number: Int) -> Int {
    return number * number
}
// good
func square(of number: Int) -> Int {
    return number * number
}
```

## Change Reference to Value

Replacing a reference type with a value type (like replacing a class with a
struct in Swift), which results in better performance and safety when comparing
and copying instances.

```swift
// bad
class PointClass {
    var x: Int
    var y: Int
    init(x: Int, y: Int) {
        self.x = x
        self.y = y
    }
}
// good
struct PointStruct {
    var x: Int
    var y: Int
}
```

## Collapse Hierarchy

Eliminating unnecessary class hierarchies and simplifying the code by merging a
subclass with its superclass, or by flattening multiple class hierarchies into
one.

```swift
// bad
class BaseClass {
    func baseMethod() { }
}

class Subclass: BaseClass {
    func subclassMethod() { }
}
// good
class MergedClass {
    func baseMethod() { }
    func subclassMethod() { }
}
```

## Combine functions into Class

Grouping related functions by converting them into methods of a new or existing
class, improving code organization and modularity.

```swift
// bad
func area(width: Double, height: Double) -> Double {
    return width * height
}
func perimeter(width: Double, height: Double) -> Double {
    return 2 * (width + height)
}
// good
class Rectangle {
    var width: Double
    var height: Double

    init(width: Double, height: Double) {
        self.width = width
        self.height = height
    }
    
    func area() -> Double {
        return width * height
    }

    func perimeter() -> Double {
        return 2 * (width + height)
    }
}
```

## Combine Functions into Transform

Replacing two or more functions or closures that execute sequentially with a
single, more general one.

```swift
// bad
func double(_ number: Int) -> Int {
    return number * 2
}

func increment(_ number: Int) -> Int {
    return number + 1
}

func doubleAndIncrement(_ number: Int) -> Int {
    return increment(double(number))
}
// good
func transform(_ number: Int, with operations: [(Int) -> Int]) -> Int {
    return operations.reduce(number) { result, operation in 
        operation(result) 
    }
}
```

## Consolidate Conditional Expression

Replacing a series of nested or sequential conditional statements with a single,
unified one by using boolean operators, or by extracting the common code into a
separate function.

```swift
// bad
func isWeekend(day: Int) -> Bool {
    if day == 6 {
        return true
    } else if day == 7 {
        return true
    } else {
        return false
    }
}
// good
func isWeekend(day: Int) -> Bool {
    return day == 6 || day == 7
}
```

## Decompose Conditional

Breaking down complex conditional logic into simpler, more understandable pieces
by extracting parts of the conditional expression into separate functions with
well-defined responsibilities.

```swift
// bad
func price(isHoliday: Bool, quantity: Int) -> Double {
    if isHoliday && quantity >= 10 {
        return 0.9 * Double(quantity)
    } else {
        return Double(quantity)
    }
}
// good
func isDiscountApplicable(isHoliday: Bool, quantity: Int) -> Bool {
    return isHoliday && quantity >= 10
}

func price(isHoliday: Bool, quantity: Int) -> Double {
    return isDiscountApplicable(isHoliday: isHoliday, quantity: quantity) ? 
        0.9 * Double(quantity) : 
        Double(quantity)
}
```

## Encapsulate Collection

Restricting direct access to a collection, such as an array or dictionary, from
the outside of the class by providing appropriate methods for adding, removing,
and modifying elements in it.

```swift
// bad
class Player {
    var scores: [Int] = []
}
// good
class Player {
    private var scores: [Int] = []

    func addScore(_ score: Int) {
        scores.append(score)
    }

    func removeScore(at index: Int) {
        scores.remove(at: index)
    }

    func updateScore(_ score: Int, at index: Int) {
        scores[index] = score
    }
}
```

## Encapsulate Record (Replace Record with Data Class)

Replacing a raw data record with a class that contains methods for working with
the data. This increases encapsulation and allows for better code organization.

```swift
// bad
typealias PersonRecord = (name: String, age: Int)
// good
class Person {
    let name: String
    let age: Int

    init(name: String, age: Int) {
        self.name = name
        self.age = age
    }

    func description() -> String {
        return "Name: \(name), Age: \(age)"
    }
}
```

## Encapsulate Variable (Encapsulate Field • Self-Encapsulate Field)

Providing **getter** and **setter** methods instead of direct access to a class
variable, allowing for better control over access and modification, as well as
improving encapsulation.

```swift
// bad
class Circle {
    var radius: Double
}
// good
class Circle {
    private var _radius: Double

    init(radius: Double) {
        self._radius = radius
    }

    var radius: Double {
        get { return _radius }
        set { _radius = newValue }
    }
}
```

## Extract Class

Extract Class refactoring involves creating a new class to separate the
responsibilities of an existing one. This is useful for breaking down large
classes into smaller, more focused units.

```swift
// bad
class Employee {
    var name: String
    var address: String
    var officeNumber: String

    init(name: String, address: String, officeNumber: String) {
        self.name = name
        self.address = address
        self.officeNumber = officeNumber
    }
}
// good
class Address {
    var street: String
    var city: String
    var state: String
    var zip: String
}

class Employee {
    var name: String
    var address: Address
    var officeNumber: String

    init(name: String, address: Address, officeNumber: String) {
        self.name = name
        self.address = address
        self.officeNumber = officeNumber
    }
}
```

## Extract Function

Extract Function refactoring involves taking a part of an existing function and
moving it into a new function. This is useful for improving code readability and
organizing logic into smaller, more focused units.

```swift
// bad
func printSpecificUserInfo(user: User) {
    let name = user.name
    let age = user.age
    let formattedAge = "Age: \(age)"
    print("\(name) - \(formattedAge)")
}

// good
func formatAge(age: Int) -> String {
    return "Age: \(age)"
}

func printSpecificUserInfo(user: User) {
    let name = user.name
    let age = user.age
    let formattedAge = formatAge(age: age)
    print("\(name) - \(formattedAge)")
}
```

## Extract Method

Extract Method is the same as Extract Function but with the difference that it
operates on class or struct methods.

```swift
// bad
class Customer {
    var orders: [Order]

    func getTotalAmount() -> Double {
        var amount = 0.0
        for order in orders {
            amount += order.price
        }
        print("Total: \(amount)")

        print("\(orders.count) orders")

        return amount
    }
}

// good
class Customer {
    var orders: [Order]

    func getTotalAmount() -> Double {
        let amount = calculateTotalAmount()
        printTotalAmount(amount)

        printOrderCount()

        return amount
    }

    private func calculateTotalAmount() -> Double {
        var amount = 0.0
        for order in orders {
            amount += order.price
        }
        return amount
    }

    private func printTotalAmount(_ amount: Double) {
        print("Total: \(amount)")
    }

    private func printOrderCount() {
        print("\(orders.count) orders")
    }
}
```

## Extract Superclass

Extract Superclass is a refactoring technique used to create a new superclass
(generalization) for two or more classes that have similar features. This
superclass should contain common attributes and behaviors.

```swift
// bad
class Manager {
    var name: String
    var salary: Double
    func calculateBonus() -> Double { ... }
}

class Worker {
    var name: String
    var salary: Double
    func calculateBonus() -> Double { ... }
}

// good
class Employee {
    var name: String
    var salary: Double
    func calculateBonus() -> Double { ... }
}

class Manager: Employee { ... }
class Worker: Employee { ... }
```

## Extract Variable (Introduce Explaining Variable)

Extracting a variable means declaring a new variable to hold the value of a
complex expression to improve readability and maintainability.

```swift
// bad
let timeDifference = someEvent.endTime.timeIntervalSince(someEvent.startTime) / 3600

// good
let secondsInHour = 3600
let timeDifference = someEvent.endTime.timeIntervalSince(someEvent.startTime) / secondsInHour
```

## Hide Delegate

Hide Delegate is a refactoring method that involves encapsulating complex
navigation paths, so the client codes do not need to know about the details of
the delegate classes.

```swift
// bad
let managerName = employee.department.manager.name

// good
// in Employee class:
func getManagerName() -> String {
    return department.manager.name
}

// in client code:
let managerName = employee.getManagerName()
```

## Inline Class

Inline Class is a refactoring method used when a class isn't doing enough to
justify its existence. The class is collapsed by moving its methods and fields
into another class.

```swift
// bad
class Address {
    var street: String
    var city: String
}

class Person {
    var name: String
    var address: Address
}

// good
class Person {
    var name: String
    var street: String
    var city: String
}
```

## Inline Function (Inline Method)

Inlining a function means replacing its calls with its content.

```swift
// bad
func square(value: Int) -> Int {
    return value * value
}

let result = square(value: 5)

// good
let result = 5 * 5
```


## Inline Variable

Inlining a variable means replacing its uses with its assigned value if the
variable is only assigned once.

```swift
// bad
let tax = 0.08 * subtotal
let total = subtotal + tax

// good
let total = subtotal + (0.08 * subtotal)
```

## Inline Temp

Inline temp is a technique to eliminate unnecessary variables by replacing their
usage with the direct expression.

```swift
// bad
let basePrice = quantity * itemPrice
let discount = basePrice > 1000 ? basePrice * 0.05 : 0
let finalPrice = basePrice - discount

// good
let finalPrice = quantity * itemPrice -
                 ((quantity * itemPrice) > 1000 ? (quantity * itemPrice) * 0.05 : 0)
```


## Introduce Assertion

Introduce assertion is the process of adding assert statements within functions
to specify the expected conditions and behaviors.

```swift
// bad
func calculateArea(width: Double, height: Double) -> Double {
    let area = width * height
    return area
}

// good
func calculateArea(width: Double, height: Double) -> Double {
    assert(width >= 0, "Width must be non-negative")
    assert(height >= 0, "Height must be non-negative")

    let area = width * height
    return area
}
```

## Introduce Parameter Object

Introduce Parameter Object is a technique in which you create a single object
that groups together several parameters being passed to a function.


```swift
// bad
func calculateDistance(x1: Double, y1: Double, x2: Double, y2: Double) -> Double { ... }

// good
struct Point {
    let x: Double
    let y: Double
}

func calculateDistance(p1: Point, p2: Point) -> Double { ... }
```

## Introduce Special Case (Introduce Null Object)

Introduce Special Case is a refactoring technique used to create a subclass for
a special case, so the conditional logic can be replaced with polymorphism.

```swift
// bad
class Customer {
    var name: String?
    var isActive: Bool

    func specialCondition() -> Bool {
        return name == "unknown" || !isActive
    }
}

// good
class Customer {
    var name: String
    var isActive: Bool

    func specialCondition() -> Bool { return false }
}

class UnknownCustomer: Customer {
    override func specialCondition() -> Bool { return true }
}

class InactiveCustomer: Customer {
    override func specialCondition() -> Bool { return true }
}
```

## Move Field

Move Field is a technique used to reassign a class's fields to another class
when a field is used more by another class or should belong to another class.

```swift
// bad
class Customer {
    var credit: Double
}

class Order {
    var customer: Customer
}

// good
class Customer { }

class Order {
    var customer: Customer
    var credit: Double
}
```

## Move Function (Move Method)

Move Function is a refactoring technique when you move a method closer to the
class or object it actually uses.

```swift
// bad
class Account {
    var balance: Double

    func chargeFee(amount: Double) {
        balance -= amount
    }
}

class Bank {
    func applyMonthlyFee(account: Account) {
        account.chargeFee(amount: 15)
    }
}

// good
class Account {
    var balance: Double
    
    func applyMonthlyFee() {
        balance -= 15
    }
}

class Bank {
    func applyMonthlyFee(account: Account) {
        account.applyMonthlyFee()
    }
}
```

## Move Statements into Function

Move Statements into Function is a refactoring technique that involves moving a
code block into a separate function when the code is repeated multiple times.

```swift
// bad
func createUser() {
    // some create user logic
    print("User created")
}

func createAdmin() {
    // some create admin logic
    print("User created")
}

// good
func printUserCreatedMessage() {
    print("User created")
}

func createUser() {
    // some create user logic
    printUserCreatedMessage()
}

func createAdmin() {
    // some create admin logic
    printUserCreatedMessage()
}
```

## Move Statements to Callers

This technique is used when the same function is called with different arguments
but has different behaviors. You create separate functions for different
behaviors and move the statements to the callers, who now call the correct
function.

```swift
// bad
func setValue(name: String, value: String) {
    if name == "username" {
        UserDefaults.standard.set(value, forKey: "username")
    } else if name == "email" {
        UserDefaults.standard.set(value, forKey: "email")
    }
}

setValue(name: "username", value: "JohnDoe")
setValue(name: "email", value: "john.doe@example.com")

// good
func setUsername(value: String) {
    UserDefaults.standard.set(value, forKey: "username")
}

func setEmail(value: String) {
    UserDefaults.standard.set(value, forKey: "email")
}

setUsername(value: "JohnDoe")
setEmail(value: "john.doe@example.com")
```

## Parameterize Function (Parameterize Method)

Parameterize Function is a method for removing duplicate code by turning several
similar functions into a single function that takes a parameter to decide what
action to perform.

```swift
// bad
func addOne(value: Int) -> Int {
    return value + 1
}

func addTwo(value: Int) -> Int {
    return value + 2
}

// good
func add(value: Int, addend: Int) -> Int {
    return value + addend
}

let result1 = add(value: 5, addend: 1)
let result2 = add(value: 5, addend: 2)
```

## Preserve Whole Object

Instead of passing several data items from an object as method parameters, send
the whole object instead. This reduces the amount of parameters in the method
and improves readability.

```swift
// bad
func getTemperature(low: Int, high: Int) -> Int {
    return (low + high) / 2
}

let temperatureRange = (low: 30, high: 50)
let averageTemperature = getTemperature(low: temperatureRange.low, high: temperatureRange.high)

// good
func getTemperature(temperatureRange: (low: Int, high: Int)) -> Int {
    return (temperatureRange.low + temperatureRange.high) / 2
}

let temperatureRange = (low: 30, high: 50)
let averageTemperature = getTemperature(temperatureRange: temperatureRange)
```

## Pull Up Constructor Body

Move the constructor's initialization code from a subclass to a superclass when
the same code is present in all subclasses.

```swift
// bad
class Animal {
    var name: String
    
    init(name: String) {
        self.name = name
    }
}

class Dog: Animal {
    var breed: String

    init(name: String, breed: String) {
        self.breed = breed
        super.init(name: name)
    }
}

class Cat: Animal {
    var color: String

    init(name: String, color: String) {
        self.color = color
        super.init(name: name)
    }
}

// good
class Animal {
    var name: String
    var descriptor: String

    init(name: String, descriptor: String) {
        self.name = name
        self.descriptor = descriptor
    }
}

class Dog: Animal {
    init(name: String, breed: String) {
        super.init(name: name, descriptor: breed)
    }
}

class Cat: Animal {
    init(name: String, color: String) {
        super.init(name: name, descriptor: color)
    }
}
```

## Pull Up Field

Move a field from a subclass to a superclass when it is used by multiple
subclasses or can be useful for all subclasses.

```swift
// bad
class Person {
}

class Employee: Person {
    var salary: Double
}

class Manager: Person {
    var salary: Double
}

// good
class Person {
    var salary: Double
}

class Employee: Person {
}

class Manager: Person {
}
```

## Pull Up Method

Move a method from a subclass to a superclass if the logic is the same for all
subclasses. This prevents code duplication.

```swift
// bad
class Animal {
}

class Dog: Animal {
    func makeSound() {
        print("Bark!")
    }
}

class Cat: Animal {
    func makeSound() {
        print("Meow!")
    }
}

// good
class Animal {
    func makeSound() {
        print("Generic animal sound")
    }
}

class Dog: Animal {
    override func makeSound() {
        print("Bark!")
    }
}

class Cat: Animal {
    override func makeSound() {
        print("Meow!")
    }
}
```

## Push Down Field

Move a field from a superclass to a subclass if it is only relevant for a
specific subclass.

```swift
// bad
class Person {
    var salary: Double?
}

class Employee: Person {
}

class Manager: Person {
}

// good
class Person {
}

class Employee: Person {
    var salary: Double
}

class Manager: Person {
    var salary: Double
}
```

## Push Down Method

Move a method from a superclass to a subclass if it is only relevant for a
specific subclass or if the method's logic is different for each subclass.

```swift
// bad
class Animal {
    func makeSound() {
        print("Generic animal sound")
    }
}

class Dog: Animal {
}

class Cat: Animal {
}

// good
class Animal {
}

class Dog: Animal {
    func makeSound() {
        print("Bark!")
    }
}

class Cat: Animal {
    func makeSound() {
        print("Meow!")
    }
}
```

## Remove Dead Code

Remove code that is not used or reachable anymore. Doing so improves
maintainability and readability.

```swift
// bad
func add(a: Int, b: Int) -> Int {
    return a + b
}

func subtract(a: Int, b: Int) -> Int {
    return a - b
}

let result = add(a: 5, b: 3)

// good
func add(a: Int, b: Int) -> Int {
    return a + b
}

let result = add(a: 5, b: 3)
```

## Remove Flag Argument (Replace Parameter with Explicit Methods)

Replace a method with multiple explicit methods, each with its own purpose, when
a method uses a flag parameter to determine its behavior. This improves
readability and maintainability.

```swift
// bad
enum Shape {
    case circle
    case rectangle
}

func calculateArea(radius: Double?, width: Double?, height: Double?, shape: Shape) -> Double {
    switch shape {
    case .circle:
        return Double.pi * (radius! * radius!)
    case .rectangle:
        return width! * height!
    }
}

let circleArea = calculateArea(radius: 5, width: nil, height: nil, shape: .circle)
let rectangleArea = calculateArea(radius: nil, width: 10, height: 5, shape: .rectangle)

// good
func calculateCircleArea(radius: Double) -> Double {
    return Double.pi * (radius * radius)
}

func calculateRectangleArea(width: Double, height: Double) -> Double {
    return width * height
}

let circleArea = calculateCircleArea(radius: 5)
let rectangleArea = calculateRectangleArea(width: 10, height: 5)
```

## Remove Middle Man

Remove unnecessary delegating methods and call the destination method directly.
This reduces complexity and improves readability.

```swift
// bad
class Dog {
    func bark() {
        print("Woof!")
    }
}

class DogOwner {
    var dog: Dog

    init(dog: Dog) {
        self.dog = dog
    }

    func makeDogBark() {
        dog.bark()
    }
}

let dog = Dog()
let dogOwner = DogOwner(dog: dog)
dogOwner.makeDogBark()

// good
class Dog {
    func bark() {
        print("Woof!")
    }
}

class DogOwner {
    var dog: Dog

    init(dog: Dog) {
        self.dog = dog
    }
}

let dog = Dog()
let dogOwner = DogOwner(dog: dog)
dogOwner.dog.bark()
```

## Remove Setting Method

Remove a setting method if a field should not be mutable or setting it can lead
to inconsistencies.

```swift
// bad
class Rectangle {
    var width: Double
    var height: Double

    init(width: Double, height: Double) {
        self.width = width
        self.height = height
    }

    func setWidth(newWidth: Double) {
        width = newWidth
    }

    func setHeight(newHeight: Double) {
        height = newHeight
    }
}

// good
class Rectangle {
    let width: Double
    let height: Double

    init(width: Double, height: Double) {
        self.width = width
        self.height = height
    }
}
```

## Remove Subclass (Replace Subclass with Fields)

Replace subclasses that only have different field values or methods with a class
that has fields with multiple values. This reduces code duplication and
complexity.

```swift
// bad
class Animal {
}

class Dog: Animal {
    func getSound() -> String {
        return "Woof!"
    }
}

class Cat: Animal {
    func getSound() -> String {
        return "Meow!"
    }
}

// good
class Animal {
    let sound: String

    init(sound: String) {
        self.sound = sound
    }

    func getSound() -> String {
        return sound
    }
}

let dog = Animal(sound: "Woof!")
let cat = Animal(sound: "Meow!")
```

## Rename Field

Rename a field to make its purpose and use more clear.

```swift
// bad
class Rectangle {
    var a: Double
    var b: Double

    init(a: Double, b: Double) {
        self.a = a
        self.b = b
    }
}

// good
class Rectangle {
    var width: Double
    var height: Double

    init(width: Double, height: Double) {
        self.width = width
        self.height = height
    }
}
```

## Rename Variable

Rename a variable to make its purpose and use more clear.

```swift
// bad
let p = 3.14

// good
let pi = 3.14
```

## Replace Command with Function

Replace a command pattern object with a simple function when its sole purpose is
to execute a single action without storing any state.

```swift
// bad
class PrintCommand {
    func execute(text: String) {
        print(text)
    }
}

let printCommand = PrintCommand()
printCommand.execute(text: "Hello, world!")

// good
func printText(text: String) {
    print(text)
}

printText(text: "Hello, world!")
```

## Replace Conditional with Polymorphism

Instead of using conditionals to map different behavior for various cases,
leverage inheritance and polymorphism to achieve the same goal in a more
readable and maintainable manner. 

```swift
// bad
class Animal {
    var type: String

    init(type: String) {
        self.type = type
    }

    func makeNoise() {
        if type == "Dog" {
            print("Woof!")
        } else if type == "Cat" {
            print("Meow!")
        }
    }
}

// good
class Animal {
    func makeNoise() {
        // implementation for parent class
    }
}

class Dog: Animal {
    override func makeNoise() {
        print("Woof!")
    }
}

class Cat: Animal {
    override func makeNoise() {
        print("Meow!")
    }
}
```

## Replace Constructor with Factory Function (Replace Constructor with Factory Method)

Encapsulate the process of object creation into a separate factory function or
method instead of directly invoking constructors

```swift
// bad
class Car {
    var model: String
    var color: String

    init(model: String, color: String) {
        self.model = model
        self.color = color
    }
}

let car1 = Car(model: "Model1", color: "Red")

// good
class Car {
    var model: String
    var color: String

    private init(model: String, color: String) {
        self.model = model
        self.color = color
    }

    static func createCar(model: String, color: String) -> Car {
        return Car(model: model, color: color)
    }
}

let car1 = Car.createCar(model: "Model1", color: "Red")
```

## Replace Control Flag with Break (Remove Control Flag)

Instead of using a control flag variable to control loop exit, use the 'break'
statement.

```swift
// bad
func findTarget(numbers: [Int], target: Int) -> Bool {
    var found = false
    for number in numbers {
        if number == target {
            found = true
            break
        }
    }
    return found
}

// good
func findTarget(numbers: [Int], target: Int) -> Bool {
    for number in numbers {
        if number == target {
            return true
        }
    }
    return false
}
```


## Replace Derived Variable with Query

Instead of using a derived variable that holds the result of some operation,
create a separate query function or computed property which calculates the
result on the fly. 

```swift
// bad
class Circle {
    var radius: Double
    var area: Double

    init(radius: Double) {
        self.radius = radius
        self.area = 3.14 * radius * radius
    }
}

// good
class Circle {
    var radius: Double

    init(radius: Double) {
        self.radius = radius
    }

    var area: Double {
        return 3.14 * radius * radius
    }
}
```

## Replace Error Code with Exception

Instead of using error codes to signify failure, use exceptions to make the code
more readable. 

```swift
// bad
func divide(_ a: Double, by b: Double) -> (Double?, String?) {
    if b == 0 {
        return (nil, "Division by zero")
    }
    return (a / b, nil)
}

// good
func divide(_ a: Double, by b: Double) throws -> Double {
    if b == 0 {
        throw NSError(domain: "Division by zero", code: 1, userInfo: nil)
    }
    return a / b
}
```

## Replace Exception with Precheck (Replace Exception with Test)

Instead of using exceptions as the sole method of catching errors, first test
the precondition to see if it will cause an error and then act accordingly.

```swift
// bad
func divide(_ a: Double, by b: Double) throws -> Double {
    if b == 0 {
        throw NSError(domain: "Division by zero", code: 1, userInfo: nil)
    }
    return a / b
}

do {
    try divide(4, by: 0)
} catch {
    print("Error occurred")
}

// good
func divide(_ a: Double, by b: Double) -> Double? {
    if b == 0 {
        return nil
    }
    return a / b
}

if let result = divide(4, by: 0) {
    print(result)
} else {
    print("Error occurred")
}
```


## Replace Function with Command (Replace Method with Method Object)

Encapsulate a function or method into its own self-contained class, thereby
converting it into a command object.

```swift
// bad
class Account {
    var balance: Double

    init(balance: Double) {
        self.balance = balance
    }

    func transferFunds(to account: Account, amount: Double) -> Bool {
        if balance >= amount {
            balance -= amount
            account.balance += amount
            return true
        } else {
            return false
        }
    }
}

// good
class TransferFunds {
    var sourceAccount: Account
    var targetAccount: Account
    var amount: Double

    init(sourceAccount: Account, targetAccount: Account, amount: Double) {
        self.sourceAccount = sourceAccount
        self.targetAccount = targetAccount
        self.amount = amount
    }

    func execute() -> Bool {
        if sourceAccount.balance >= amount {
            sourceAccount.balance -= amount
            targetAccount.balance += amount
            return true
        } else {
            return false
        }
    }
}
```

## Replace Inline Code with Function Call

Instead of using inline code to perform an operation, call a function or method
from within the code block to improve readability and reusability.

```swift
// bad
let numbers = [1, 2, 3, 4, 5]
var doubledNumbers = [Int]()
for number in numbers {
    doubledNumbers.append(number * 2)
}

// good
func double(_ number: Int) -> Int {
    return number * 2
}

let doubledNumbers = numbers.map { double($0) }
```

## Replace Loop with Pipeline

Instead of using loops, use functional programming concepts like 'map',
'filter', or 'reduce' to transform a collection of data into a new form.

```swift
// bad
let numbers = [1, 2, 3, 4, 5]
var sumOfSquares = 0
for number in numbers {
    sumOfSquares += number * number
}

// good
let sumOfSquares = numbers.reduce(0) { 
    $0 + ($1 * $1) 
}
```

## Replace Magic Literal (Replace Magic Number with Symbolic Constant)

Instead of using a magic literal or magic number directly in the code, assign it
to a meaningful constant and use the constant instead.

```swift
// bad
let circleArea = 3.14 * radius * radius

// good
let pi = 3.14
let circleArea = pi * radius * radius
```

## Replace Nested Conditional with Guard Clauses

Instead of nesting conditional statements, write each condition as a separate
guard clause to make the code more readable.

```swift
// bad
func process(number: Int) {
    if number > 0 {
        if number < 10 {
            print("Number is between 1 and 9")
        } else {
            print("Number is greater than 9")
        }
    } else {
        print("Number is less than or equal to 0")
    }
}

// good
func process(number: Int) {
    guard number > 0 else {
        print("Number is less than or equal to 0")
        return
    }

    guard number < 10 else {
        print("Number is greater than 9")
        return
    }

    print("Number is between 1 and 9")
}
```

## Replace Parameter with Query (Replace Parameter with Method)

Instead of passing a parameter to a function or method, call a function,
computed property, or method from within the method or function to obtain the
needed value.

```swift
// bad
class Rectangle {
    var width: Double
    var height: Double

    init(width: Double, height: Double) {
        self.width = width
        self.height = height
    }

    func area(width:	Double, height: Double) -> Double {
        return width * height
    }
}

let rectangle = Rectangle(width: 10, height: 20)
let area = rectangle.area(width: rectangle.width, height: rectangle.height)

// good
class Rectangle {
    var width: Double
    var height: Double

    init(width: Double, height: Double) {
        self.width = width
        self.height = height
    }

    func area() -> Double {
        return width * height
    }
}

let rectangle = Rectangle(width: 10, height: 20)
let area = rectangle.area()
```

## Replace Primitive with Object (Replace Data Value with Object • Replace Type Code with Class)

Instead of using primitive data types, encapsulate data and related methods in a
class or structure to promote code reuse and maintainability.

```swift
// bad
struct Car {
    var make: String
    var model: String
    var year: Int
}

// good
class Car {
    var make: Make
    var model: Model
    var year: Year

    init(make: Make, model: Model, year: Year) {
        self.make = make
        self.model = model
        self.year = year
    }
}

class Make {
    var name: String

    init(name: String) {
        self.name = name
    }
}

class Model {
    var name: String

    init(name: String) {
        self.name = name
    }
}

class Year {
    var value: Int

    init(value: Int) {
        self.value = value
    }
}
```

## Replace Query with Parameter

When a method performs a query that can be parameterized, pass the required data
directly into the method as a parameter instead of asking for the data to remove
the dependency on the query.

```swift
// Bad
func findProducts() -> [Product] {
    let category = getCategory()
    return products.filter { $0.category == category }
}

// Good
func findProducts(in category: Category) -> [Product] {
    return products.filter { $0.category == category }
}
```

## Replace Subclass with Delegate

When you have many subclasses that add or override methods with identical code,
you can replace the subclasses with a delegate using a unified interface.

```swift
// Bad
class A {
}

class B: A {
}

class C: A {
}

// Good
protocol ADelegate {
}

class A {
    var delegate: ADelegate?
}

class B: ADelegate {
}

class C: ADelegate {
}
```

## Replace Superclass with Delegate (Replace Inheritance with Delegation)

Instead of using inheritance to share behavior, delegate these responsibilities
to another class through composition. This reduces the dependency and complexity
of inheritance.

```swift
// Bad
class Animal {
    func speak() { }
}

class Dog: Animal {
    override func speak() {
        print("Woof!")
    }
}

// Good
protocol Speaker {
    func speak()
}

class Dog: Speaker {
    func speak() {
        print("Woof!")
    }
}
```

## Replace Temp with Query

Instead of storing the result of an expression (query) in a temporary variable,
consider extracting the expression into a method.

```swift
// Bad
func getTotalPrice() -> Double {
    let basePrice = quantity * itemPrice
    return basePrice * discountFactor
}

// Good
func getTotalPrice() -> Double {
    return basePrice * discountFactor
}

func basePrice() -> Double {
    return quantity * itemPrice
}
```

## Replace Type Code with Subclasses (Extract Subclass • Replace Type Code with State/Strategy)

If a class has a type code that affects the behavior of the object, you can
replace the type code with subclasses representing each type code, and behavior
variations can then be implemented in the subclasses.

```swift
// Bad
enum EmployeeType {
    case engineer, manager
}

class Employee {
    var type: EmployeeType
}

// Good
class Employee {
}

class Engineer: Employee {
}

class Manager: Employee {
}
```

## Return Modified Value

When a method modifies the value of an input parameter, consider returning the
modified value instead of modifying the original value.

```swift
// Bad
func increment(_ value: inout Int) {
    value += 1
}

// Good
func increment(_ value: Int) -> Int {
    return value + 1
}
```

## Separate Query from Modifier

Do not mix queries (methods that return data) with modifiers (methods that
change the object’s state) in the same method. Split them into two separate
methods to make the code more readable and maintainable.

```swift
// Bad
func getSpeedAndApplyBrakes() -> Int {
    speed -= 10
    return speed
}

// Good
func getSpeed() -> Int {
    return speed
}

func applyBrakes() {
    speed -= 10
}
```

## Slide Statements (Consolidate Duplicate Conditional Fragments)

Move statements that are related to each other closer together. This helps in
understanding the code and maintaining it. Also, if the same statements are
repeatedly written within conditional branches, move them outside the branches.

```swift
// Bad
if condition {
    doSomething()
    doSomethingElse()
    doThirdThing()
} else {
    doSomething()
    doAnotherThing()
    doThirdThing()
}

// Good
doSomething()
if condition {
    doSomethingElse()
} else {
    doAnotherThing()
}
doThirdThing()
```

## Split Loop

When a loop does multiple tasks, consider splitting it into multiple loops with
one task per loop. This makes the code easier to read and maintain.

```swift
// Bad
for product in products {
    updatePrice(for: product)
    updateStock(for: product)
}

// Good
for product in products {
    updatePrice(for: product)
}

for product in products {
    updateStock(for: product)
}
```

## Split Phase

When a code segment does different tasks in sequence, split the tasks into
separate methods, where each method does only one task. This makes the code more
maintainable and easier to understand.

```swift
// Bad
func processOrders() {
    // Step 1: parse order data
    // Step 2: validate order data
    // Step 3: persist orders to database
}

// Good
func processOrders() {
    parseOrderData()
    validateOrderData()
    persistOrdersToDatabase()
}
```

## Split Variable (Remove Assignments to Parameters • Split Temp)

When a variable holds different values for different purposes, create separate
variables for each purpose. This improves code readability and maintainability.

```swift
// Bad
var distance = calculateDistance()
distance = distance * 0.621371

// Good
let distanceInKm = calculateDistance()
let distanceInMiles = distanceInKm * 0.621371
```

## Substitute Algorithm

If a more efficient algorithm exists for solving a particular problem, then
replace the existing algorithm with the more efficient one.

```swift
// Bad
func findDuplicates(in values: [Int]) -> [Int] {
    var duplicates = [Int]()
    for value in values {
        if values.filter({ $0 == value }).count > 1 {
            duplicates.append(value)
        }
    }
    return duplicates
}

// Good
func findDuplicates(in values: [Int]) -> [Int] {
    var countDict = [Int: Int]()
    for value in values {
        countDict[value] = (countDict[value] ?? 0) + 1
    }
    return countDict.filter { $0.value > 1 }.map { $0.key }
}
```
