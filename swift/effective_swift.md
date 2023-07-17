- [Materials](#materials)
- [Creating and Destroying Objects](#creating-and-destroying-objects)
  - [item1: Consider static factory methods instead of constructors](#item1-consider-static-factory-methods-instead-of-constructors)
  - [item2: Consider a builder when faced with many constructor parameters](#item2-consider-a-builder-when-faced-with-many-constructor-parameters)
  - [item3: Enforce the singleton property with a private constructor or an enum type](#item3-enforce-the-singleton-property-with-a-private-constructor-or-an-enum-type)
  - [item4: Enforce noninstantiability with a private constructor](#item4-enforce-noninstantiability-with-a-private-constructor)
  - [item5: Prefer dependency injection to hardwiring resources](#item5-prefer-dependency-injection-to-hardwiring-resources)
  - [item6: Avoid creating unnecessary objects](#item6-avoid-creating-unnecessary-objects)
  - [item7: Eliminate obsolete object references](#item7-eliminate-obsolete-object-references)
  - [item8: Avoid finalizers and cleaners](#item8-avoid-finalizers-and-cleaners)
  - [item9: Prefer try-with-resources to try-finally](#item9-prefer-try-with-resources-to-try-finally)
- [Methods Common to All Objects](#methods-common-to-all-objects)
  - [item10: Obey the general contract when overriding equals](#item10-obey-the-general-contract-when-overriding-equals)
  - [item11: Always override hashCode when you override equals](#item11-always-override-hashcode-when-you-override-equals)
  - [item12: Always override toString](#item12-always-override-tostring)
  - [item13: Override clone judiciously](#item13-override-clone-judiciously)
  - [item14: Consider implementing Comparable](#item14-consider-implementing-comparable)
- [Classes and Interfaces](#classes-and-interfaces)
  - [item15: Minimize the accessibility of classes and members](#item15-minimize-the-accessibility-of-classes-and-members)
  - [item16: In public classes, use accessor methods, not public fields](#item16-in-public-classes-use-accessor-methods-not-public-fields)
  - [item17: Minimize mutability](#item17-minimize-mutability)
  - [item18: Favor composition over inheritance](#item18-favor-composition-over-inheritance)
  - [item19: Design and document for inheritance or else prohibit it](#item19-design-and-document-for-inheritance-or-else-prohibit-it)
  - [item20: Prefer interfaces to abstract classes](#item20-prefer-interfaces-to-abstract-classes)
  - [item21: Design interfaces for posterity](#item21-design-interfaces-for-posterity)
  - [item22: Use interfaces only to define types](#item22-use-interfaces-only-to-define-types)
  - [item23: Prefer class hierarchies to tagged classes](#item23-prefer-class-hierarchies-to-tagged-classes)
  - [item24: Favor static member classes over nonstatic](#item24-favor-static-member-classes-over-nonstatic)
  - [item25: Limit source files to a single top-level class](#item25-limit-source-files-to-a-single-top-level-class)
- [Generics](#generics)
  - [item26: Don’t use raw types](#item26-dont-use-raw-types)
  - [item27: Eliminate unchecked warnings](#item27-eliminate-unchecked-warnings)
  - [item28: Prefer lists to arrays](#item28-prefer-lists-to-arrays)
  - [item29: Favor generic types](#item29-favor-generic-types)
  - [item30: Favor generic methods](#item30-favor-generic-methods)
  - [item31: Use bounded wildcards to increase API flexibility](#item31-use-bounded-wildcards-to-increase-api-flexibility)
  - [item32: Combine generics and varargs judiciously](#item32-combine-generics-and-varargs-judiciously)
  - [item33: Consider typesafe heterogeneous containers](#item33-consider-typesafe-heterogeneous-containers)
- [Enums and Annotations](#enums-and-annotations)
  - [item34: Use enums instead of int constants](#item34-use-enums-instead-of-int-constants)
  - [item35: Use instance fields instead of ordinals](#item35-use-instance-fields-instead-of-ordinals)
  - [item36: Use EnumSet instead of bit fields](#item36-use-enumset-instead-of-bit-fields)
  - [item37: Use EnumMap instead of ordinal indexing](#item37-use-enummap-instead-of-ordinal-indexing)
  - [item38: Emulate extensible enums with interfaces](#item38-emulate-extensible-enums-with-interfaces)
  - [item39: Prefer annotations to naming patterns](#item39-prefer-annotations-to-naming-patterns)
  - [item40: Consistently use the Override annotation](#item40-consistently-use-the-override-annotation)
  - [item41: Use marker interfaces to define types](#item41-use-marker-interfaces-to-define-types)
- [Lambdas and Streams](#lambdas-and-streams)
  - [item42: Prefer lambdas to anonymous classes](#item42-prefer-lambdas-to-anonymous-classes)
  - [item43: Prefer method references to lambdas](#item43-prefer-method-references-to-lambdas)
  - [item44: Favor the use of standard functional interfaces](#item44-favor-the-use-of-standard-functional-interfaces)
  - [item45: Use streams judiciously](#item45-use-streams-judiciously)
  - [item46: Prefer side-effect-free functions in streams](#item46-prefer-side-effect-free-functions-in-streams)
  - [item47: Prefer Collection to Stream as a return type](#item47-prefer-collection-to-stream-as-a-return-type)
  - [item48: Use caution when making streams parallel](#item48-use-caution-when-making-streams-parallel)
- [Methods](#methods)
  - [item49: Check parameters for validity](#item49-check-parameters-for-validity)
  - [item50: Make defensive copies when needed](#item50-make-defensive-copies-when-needed)
  - [item51: Design method signatures carefully](#item51-design-method-signatures-carefully)
  - [item52: Use overloading judiciously](#item52-use-overloading-judiciously)
  - [item53: Use varargs judiciously](#item53-use-varargs-judiciously)
  - [item54: Return empty collections or arrays, not nulls](#item54-return-empty-collections-or-arrays-not-nulls)
  - [item55: Return optionals judiciously](#item55-return-optionals-judiciously)
  - [item56: Write doc comments for all exposed API elements](#item56-write-doc-comments-for-all-exposed-api-elements)
- [General Programming](#general-programming)
  - [item57: Minimize the scope of local variables](#item57-minimize-the-scope-of-local-variables)
  - [item58: Prefer for-each loops to traditional for loops](#item58-prefer-for-each-loops-to-traditional-for-loops)
  - [item59: Know and use the libraries](#item59-know-and-use-the-libraries)
  - [item60: Avoid float and double if exact answers are required](#item60-avoid-float-and-double-if-exact-answers-are-required)
  - [item61: Prefer primitive types to boxed primitives](#item61-prefer-primitive-types-to-boxed-primitives)
  - [item62: Avoid strings where other types are more appropriate](#item62-avoid-strings-where-other-types-are-more-appropriate)
  - [item63: Beware the performance of string concatenation](#item63-beware-the-performance-of-string-concatenation)
  - [item64: Refer to objects by their interfaces](#item64-refer-to-objects-by-their-interfaces)
  - [item65: Prefer interfaces to reflection](#item65-prefer-interfaces-to-reflection)
  - [item66: Use native methods judiciously](#item66-use-native-methods-judiciously)
  - [item67: Optimize judiciously](#item67-optimize-judiciously)
  - [item68: Adhere to generally accepted naming conventions](#item68-adhere-to-generally-accepted-naming-conventions)
- [Exceptions](#exceptions)
  - [item69: Use exceptions only for exceptional conditions](#item69-use-exceptions-only-for-exceptional-conditions)
  - [item70: Use checked exceptions for recoverable conditions and runtime exceptions for programming errors](#item70-use-checked-exceptions-for-recoverable-conditions-and-runtime-exceptions-for-programming-errors)
  - [item71: Avoid unnecessary use of checked exceptions](#item71-avoid-unnecessary-use-of-checked-exceptions)
  - [item72: Favor the use of standard exceptions](#item72-favor-the-use-of-standard-exceptions)
  - [item73: Throw exceptions appropriate to the abstraction](#item73-throw-exceptions-appropriate-to-the-abstraction)
  - [item74: Document all exceptions thrown by each method](#item74-document-all-exceptions-thrown-by-each-method)
  - [item75: Include failure-capture information in detail messages](#item75-include-failure-capture-information-in-detail-messages)
  - [item76: Strive for failure atomicity](#item76-strive-for-failure-atomicity)
  - [item77: Don’t ignore exceptions](#item77-dont-ignore-exceptions)
- [Concurrency](#concurrency)
  - [item78: Synchronize access to shared mutable data](#item78-synchronize-access-to-shared-mutable-data)
  - [item79: Avoid excessive synchronization](#item79-avoid-excessive-synchronization)
  - [item80: Prefer executors, tasks, and streams to threads](#item80-prefer-executors-tasks-and-streams-to-threads)
  - [item81: Prefer concurrency utilities to wait and notify](#item81-prefer-concurrency-utilities-to-wait-and-notify)
  - [item82: Document thread safety](#item82-document-thread-safety)
  - [item83: Use lazy initialization judiciously](#item83-use-lazy-initialization-judiciously)
  - [item84: Don’t depend on the thread scheduler](#item84-dont-depend-on-the-thread-scheduler)
- [Serialization](#serialization)
  - [item85: Prefer alternatives to Java serialization](#item85-prefer-alternatives-to-java-serialization)
  - [item86: Implement Serializable with great caution](#item86-implement-serializable-with-great-caution)
  - [item87: Consider using a custom serialized form](#item87-consider-using-a-custom-serialized-form)
  - [item88: Write readObject methods defensively](#item88-write-readobject-methods-defensively)
  - [item89: For instance control, prefer enum types to readResolve](#item89-for-instance-control-prefer-enum-types-to-readresolve)
  - [item90: Consider serialization proxies instead of serialized instances](#item90-consider-serialization-proxies-instead-of-serialized-instances)

----

# Materials

* [Effective Swift](https://theswiftists.github.io/effective-swift/)
  * Effective Java 를 Swift 로 옮긴 것

# Creating and Destroying Objects
## item1: Consider static factory methods instead of constructors
## item2: Consider a builder when faced with many constructor parameters
## item3: Enforce the singleton property with a private constructor or an enum type
## item4: Enforce noninstantiability with a private constructor
## item5: Prefer dependency injection to hardwiring resources
## item6: Avoid creating unnecessary objects
## item7: Eliminate obsolete object references
## item8: Avoid finalizers and cleaners
## item9: Prefer try-with-resources to try-finally

# Methods Common to All Objects
## item10: Obey the general contract when overriding equals
## item11: Always override hashCode when you override equals
## item12: Always override toString
## item13: Override clone judiciously
## item14: Consider implementing Comparable

# Classes and Interfaces
## item15: Minimize the accessibility of classes and members
## item16: In public classes, use accessor methods, not public fields
## item17: Minimize mutability
## item18: Favor composition over inheritance
## item19: Design and document for inheritance or else prohibit it
## item20: Prefer interfaces to abstract classes
## item21: Design interfaces for posterity
## item22: Use interfaces only to define types
## item23: Prefer class hierarchies to tagged classes
## item24: Favor static member classes over nonstatic
## item25: Limit source files to a single top-level class

# Generics
## item26: Don’t use raw types
## item27: Eliminate unchecked warnings
## item28: Prefer lists to arrays
## item29: Favor generic types
## item30: Favor generic methods
## item31: Use bounded wildcards to increase API flexibility
## item32: Combine generics and varargs judiciously
## item33: Consider typesafe heterogeneous containers

# Enums and Annotations
## item34: Use enums instead of int constants
## item35: Use instance fields instead of ordinals
## item36: Use EnumSet instead of bit fields
## item37: Use EnumMap instead of ordinal indexing
## item38: Emulate extensible enums with interfaces
## item39: Prefer annotations to naming patterns
## item40: Consistently use the Override annotation
## item41: Use marker interfaces to define types

# Lambdas and Streams
## item42: Prefer lambdas to anonymous classes
## item43: Prefer method references to lambdas
## item44: Favor the use of standard functional interfaces
## item45: Use streams judiciously
## item46: Prefer side-effect-free functions in streams
## item47: Prefer Collection to Stream as a return type
## item48: Use caution when making streams parallel

# Methods
## item49: Check parameters for validity
## item50: Make defensive copies when needed
## item51: Design method signatures carefully
## item52: Use overloading judiciously
## item53: Use varargs judiciously
## item54: Return empty collections or arrays, not nulls
## item55: Return optionals judiciously
## item56: Write doc comments for all exposed API elements

# General Programming
## item57: Minimize the scope of local variables
## item58: Prefer for-each loops to traditional for loops
## item59: Know and use the libraries
## item60: Avoid float and double if exact answers are required
## item61: Prefer primitive types to boxed primitives
## item62: Avoid strings where other types are more appropriate
## item63: Beware the performance of string concatenation
## item64: Refer to objects by their interfaces
## item65: Prefer interfaces to reflection
## item66: Use native methods judiciously
## item67: Optimize judiciously
## item68: Adhere to generally accepted naming conventions

# Exceptions
## item69: Use exceptions only for exceptional conditions
## item70: Use checked exceptions for recoverable conditions and runtime exceptions for programming errors
## item71: Avoid unnecessary use of checked exceptions
## item72: Favor the use of standard exceptions
## item73: Throw exceptions appropriate to the abstraction
## item74: Document all exceptions thrown by each method
## item75: Include failure-capture information in detail messages
## item76: Strive for failure atomicity
## item77: Don’t ignore exceptions

# Concurrency
## item78: Synchronize access to shared mutable data
## item79: Avoid excessive synchronization
## item80: Prefer executors, tasks, and streams to threads
## item81: Prefer concurrency utilities to wait and notify
## item82: Document thread safety
## item83: Use lazy initialization judiciously
## item84: Don’t depend on the thread scheduler

# Serialization
## item85: Prefer alternatives to Java serialization
## item86: Implement Serializable with great caution
## item87: Consider using a custom serialized form
## item88: Write readObject methods defensively
## item89: For instance control, prefer enum types to readResolve
## item90: Consider serialization proxies instead of serialized instances
