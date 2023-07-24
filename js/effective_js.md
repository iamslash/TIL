- [Materials](#materials)
- [Accustoming Yourself to JavaScript](#accustoming-yourself-to-javascript)
  - [Know Which JavaScript You Are Using](#know-which-javascript-you-are-using)
  - [Understand JavaScript’s Floating-Point Numbers](#understand-javascripts-floating-point-numbers)
  - [Beware of Implicit Coercions](#beware-of-implicit-coercions)
  - [Prefer Primitives to Object Wrappers](#prefer-primitives-to-object-wrappers)
  - [Avoid using == with Mixed Types](#avoid-using--with-mixed-types)
  - [Learn the Limits of Semicolon Insertion](#learn-the-limits-of-semicolon-insertion)
  - [Think of Strings As Sequences of 16-Bit Code Units](#think-of-strings-as-sequences-of-16-bit-code-units)
- [Variable Scope](#variable-scope)
  - [Minimize Use of the Global Object](#minimize-use-of-the-global-object)
  - [Always Declare Local Variables](#always-declare-local-variables)
  - [Avoid with](#avoid-with)
  - [Get Comfortable with Closures](#get-comfortable-with-closures)
  - [Understand Variable Hoisting](#understand-variable-hoisting)
  - [Use Immediately Invoked Function Expressions to Create Local Scopes](#use-immediately-invoked-function-expressions-to-create-local-scopes)
  - [Beware of Unportable Scoping of Named Function Expressions](#beware-of-unportable-scoping-of-named-function-expressions)
  - [Beware of Unportable Scoping of Block-Local Function Declarations](#beware-of-unportable-scoping-of-block-local-function-declarations)
  - [Avoid Creating Local Variables with eval](#avoid-creating-local-variables-with-eval)
  - [Prefer Indirect eval to Direct eval](#prefer-indirect-eval-to-direct-eval)
- [Working with Functions](#working-with-functions)
  - [Understand the Difference between Function, Method, and Constructor Calls](#understand-the-difference-between-function-method-and-constructor-calls)
  - [Get Comfortable Using Higher-Order Functions](#get-comfortable-using-higher-order-functions)
  - [Use call to Call Methods with a Custom Receiver](#use-call-to-call-methods-with-a-custom-receiver)
  - [Use apply to Call Functions with Different Numbers of Arguments](#use-apply-to-call-functions-with-different-numbers-of-arguments)
  - [Use arguments to Create Variadic Functions](#use-arguments-to-create-variadic-functions)
  - [Never Modify the arguments Object](#never-modify-the-arguments-object)
  - [Use a Variable to Save a Reference to arguments](#use-a-variable-to-save-a-reference-to-arguments)
  - [Use bind to Extract Methods with a Fixed Receiver](#use-bind-to-extract-methods-with-a-fixed-receiver)
  - [Use bind to Curry Functions](#use-bind-to-curry-functions)
  - [Prefer Closures to Strings for Encapsulating Code](#prefer-closures-to-strings-for-encapsulating-code)
  - [Avoid Relying on the toString Method of Functions](#avoid-relying-on-the-tostring-method-of-functions)
  - [Avoid Nonstandard Stack Inspection Properties](#avoid-nonstandard-stack-inspection-properties)
- [Objects and Prototypes](#objects-and-prototypes)
  - [Understand the Difference between prototype, getPrototypeOf, and __proto__](#understand-the-difference-between-prototype-getprototypeof-and-proto)
  - [Prefer Object.getPrototypeOf to __proto__](#prefer-objectgetprototypeof-to-proto)
  - [Never Modify __proto__](#never-modify-proto)
  - [Make Your Constructors new-Agnostic](#make-your-constructors-new-agnostic)
  - [Store Methods on Prototypes](#store-methods-on-prototypes)
  - [Use Closures to Store Private Data](#use-closures-to-store-private-data)
  - [Store Instance State Only on Instance Objects](#store-instance-state-only-on-instance-objects)
  - [Recognize the Implicit Binding of this](#recognize-the-implicit-binding-of-this)
  - [Call Superclass Constructors from Subclass Constructors](#call-superclass-constructors-from-subclass-constructors)
  - [Never Reuse Superclass Property Names](#never-reuse-superclass-property-names)
  - [Avoid Inheriting from Standard Classes](#avoid-inheriting-from-standard-classes)
  - [Treat Prototypes As an Implementation Detail](#treat-prototypes-as-an-implementation-detail)
  - [Avoid Reckless Monkey-Patching](#avoid-reckless-monkey-patching)
- [Arrays and Dictionaries](#arrays-and-dictionaries)
  - [Build Lightweight Dictionaries from Direct Instances of Object](#build-lightweight-dictionaries-from-direct-instances-of-object)
  - [Use null Prototypes to Prevent Prototype Pollution](#use-null-prototypes-to-prevent-prototype-pollution)
  - [Use hasOwnProperty to Protect Against Prototype Pollution](#use-hasownproperty-to-protect-against-prototype-pollution)
  - [Prefer Arrays to Dictionaries for Ordered Collections](#prefer-arrays-to-dictionaries-for-ordered-collections)
  - [Never Add Enumerable Properties to Object.prototype](#never-add-enumerable-properties-to-objectprototype)
  - [Avoid Modifying an Object during Enumeration](#avoid-modifying-an-object-during-enumeration)
  - [Prefer for Loops to for...in Loops for Array Iteration](#prefer-for-loops-to-forin-loops-for-array-iteration)
  - [Prefer Iteration Methods to Loops](#prefer-iteration-methods-to-loops)
  - [Reuse Generic Array Methods on Array-Like Objects](#reuse-generic-array-methods-on-array-like-objects)
  - [Prefer Array Literals to the Array Constructor](#prefer-array-literals-to-the-array-constructor)
- [Library and API Design](#library-and-api-design)
  - [Maintain Consistent Conventions](#maintain-consistent-conventions)
  - [Treat undefined As “No Value”](#treat-undefined-as-no-value)
  - [Accept Options Objects for Keyword Arguments](#accept-options-objects-for-keyword-arguments)
  - [Avoid Unnecessary State](#avoid-unnecessary-state)
  - [Use Structural Typing for Flexible Interfaces](#use-structural-typing-for-flexible-interfaces)
  - [Distinguish between Array and Array-Like](#distinguish-between-array-and-array-like)
  - [Avoid Excessive Coercion](#avoid-excessive-coercion)
  - [Support Method Chaining](#support-method-chaining)
- [Concurrency](#concurrency)
  - [Don’t Block the Event Queue on I/O](#dont-block-the-event-queue-on-io)
  - [Use Nested or Named Callbacks for Asynchronous Sequencing](#use-nested-or-named-callbacks-for-asynchronous-sequencing)
  - [Be Aware of Dropped Errors](#be-aware-of-dropped-errors)
  - [Use Recursion for Asynchronous Loops](#use-recursion-for-asynchronous-loops)
  - [Don’t Block the Event Queue on Computation](#dont-block-the-event-queue-on-computation)
  - [Use a Counter to Perform Concurrent Operations](#use-a-counter-to-perform-concurrent-operations)
  - [Never Call Asynchronous Callbacks Synchronously](#never-call-asynchronous-callbacks-synchronously)
  - [Use Promises for Cleaner Asynchronous Logic](#use-promises-for-cleaner-asynchronous-logic)

-----

# Materials

* [Effective JavaScript](http://effectivejs.com/)
  * [src](https://github.com/effectivejs/code)
  
# Accustoming Yourself to JavaScript
##  Know Which JavaScript You Are Using
##  Understand JavaScript’s Floating-Point Numbers
##  Beware of Implicit Coercions
##  Prefer Primitives to Object Wrappers
##  Avoid using == with Mixed Types
##  Learn the Limits of Semicolon Insertion
##  Think of Strings As Sequences of 16-Bit Code Units
# Variable Scope
##  Minimize Use of the Global Object
##  Always Declare Local Variables
##  Avoid with
##  Get Comfortable with Closures
##  Understand Variable Hoisting
##  Use Immediately Invoked Function Expressions to Create Local Scopes
##  Beware of Unportable Scoping of Named Function Expressions
##  Beware of Unportable Scoping of Block-Local Function Declarations
##  Avoid Creating Local Variables with eval
##  Prefer Indirect eval to Direct eval
# Working with Functions
##  Understand the Difference between Function, Method, and Constructor Calls
##  Get Comfortable Using Higher-Order Functions
##  Use call to Call Methods with a Custom Receiver
##  Use apply to Call Functions with Different Numbers of Arguments
##  Use arguments to Create Variadic Functions
##  Never Modify the arguments Object
##  Use a Variable to Save a Reference to arguments
##  Use bind to Extract Methods with a Fixed Receiver
##  Use bind to Curry Functions
##  Prefer Closures to Strings for Encapsulating Code
##  Avoid Relying on the toString Method of Functions
##  Avoid Nonstandard Stack Inspection Properties
# Objects and Prototypes
##  Understand the Difference between prototype, getPrototypeOf, and __proto__
##  Prefer Object.getPrototypeOf to __proto__
##  Never Modify __proto__
##  Make Your Constructors new-Agnostic
##  Store Methods on Prototypes
##  Use Closures to Store Private Data
##  Store Instance State Only on Instance Objects
##  Recognize the Implicit Binding of this
##  Call Superclass Constructors from Subclass Constructors
##  Never Reuse Superclass Property Names
##  Avoid Inheriting from Standard Classes
##  Treat Prototypes As an Implementation Detail
##  Avoid Reckless Monkey-Patching
# Arrays and Dictionaries
##  Build Lightweight Dictionaries from Direct Instances of Object
##  Use null Prototypes to Prevent Prototype Pollution
##  Use hasOwnProperty to Protect Against Prototype Pollution
##  Prefer Arrays to Dictionaries for Ordered Collections
##  Never Add Enumerable Properties to Object.prototype
##  Avoid Modifying an Object during Enumeration
##  Prefer for Loops to for...in Loops for Array Iteration
##  Prefer Iteration Methods to Loops
##  Reuse Generic Array Methods on Array-Like Objects
##  Prefer Array Literals to the Array Constructor
# Library and API Design
##  Maintain Consistent Conventions
##  Treat undefined As “No Value”
##  Accept Options Objects for Keyword Arguments
##  Avoid Unnecessary State
##  Use Structural Typing for Flexible Interfaces
##  Distinguish between Array and Array-Like
##  Avoid Excessive Coercion
##  Support Method Chaining
# Concurrency
##  Don’t Block the Event Queue on I/O
##  Use Nested or Named Callbacks for Asynchronous Sequencing
##  Be Aware of Dropped Errors
##  Use Recursion for Asynchronous Loops
##  Don’t Block the Event Queue on Computation
##  Use a Counter to Perform Concurrent Operations
##  Never Call Asynchronous Callbacks Synchronously
##  Use Promises for Cleaner Asynchronous Logic
