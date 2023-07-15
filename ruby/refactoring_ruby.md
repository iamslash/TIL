- [Materials](#materials)
- [Catalog](#catalog)
  - [Change Function Declaration (Add Parameter • Change Signature • Remove Parameter • Rename Function • Rename Method)](#change-function-declaration-add-parameter--change-signature--remove-parameter--rename-function--rename-method)
  - [Change Reference to Value](#change-reference-to-value)
  - [Change Value to Reference](#change-value-to-reference)
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

[Refactoring Catalog | martinfowler](https://refactoring.com/catalog/)

# Catalog

## Change Function Declaration (Add Parameter • Change Signature • Remove Parameter • Rename Function • Rename Method)

```ruby
# as-is
function circum(radius) { ... }

# to-be
function circumference(radius) { ... }
```

## Change Reference to Value

```rb
# as-is
class Product {
    applyDiscount(arg) { 
        this._price.amount -= arg; 
    }
}

# to-be
class Product {
    applydiscount(arg) {
        this._price = new Money(this._price.amount - arg, this._price.currency);
    }
}
```

## Change Value to Reference

```rb
# as-is
let customer = new Customer(customerData);

# to-be
let customer = customerRepository.get(customerData.id);
```

## Collapse Hierarchy

## Combine functions into Class

## Combine Functions into Transform

## Consolidate Conditional Expression

## Decompose Conditional

## Encapsulate Collection

## Encapsulate Record (Replace Record with Data Class)

## Encapsulate Variable (Encapsulate Field • Self-Encapsulate Field)

## Extract Class
## Extract Function
## Extract Method

## Extract Superclass
## Extract Variable (Introduce Explaining Variable)

## Hide Delegate

```ruby
# as-is
manager = aPerson.department.manager;

# to-be
manager = aPerson.manager

class Person {
    get manager() { 
        return this.department.manager; 
    }
}
```

## Inline Class
## Inline Function (Inline Method)

## Inline Variable
## Inline Temp

## Introduce Assertion
## Introduce Parameter Object
## Introduce Special Case (Introduce Null Object)

## Move Field
## Move Function (Move Method)

## Move Statements into Function
## Move Statements to Callers
## Parameterize Function (Parameterize Method)

## Preserve Whole Object
## Pull Up Constructor Body
## Pull Up Field
## Pull Up Method
## Push Down Field
## Push Down Method
## Remove Dead Code
## Remove Flag Argument (Replace Parameter with Explicit Methods)

## Remove Middle Man
## Remove Setting Method
## Remove Subclass (Replace Subclass with Fields)

## Rename Field
## Rename Variable
## Replace Command with Function
## Replace Conditional with Polymorphism
## Replace Constructor with Factory Function (Replace Constructor with Factory Method)

## Replace Control Flag with Break (Remove Control Flag)

## Replace Derived Variable with Query
## Replace Error Code with Exception
## Replace Exception with Precheck (Replace Exception with Test)

## Replace Function with Command (Replace Method with Method Object)

## Replace Inline Code with Function Call
## Replace Loop with Pipeline
## Replace Magic Literal (Replace Magic Number with Symbolic Constant)

## Replace Nested Conditional with Guard Clauses
## Replace Parameter with Query (Replace Parameter with Method)

## Replace Primitive with Object (Replace Data Value with Object • Replace Type Code with Class)

## Replace Query with Parameter
## Replace Subclass with Delegate
## Replace Superclass with Delegate (Replace Inheritance with Delegation)

## Replace Temp with Query
## Replace Type Code with Subclasses (Extract Subclass • Replace Type Code with State/Strategy)

## Return Modified Value
## Separate Query from Modifier
## Slide Statements (Consolidate Duplicate Conditional Fragments)

## Split Loop
## Split Phase
## Split Variable (Remove Assignments to Parameters • Split Temp)

## Substitute Algorithm
