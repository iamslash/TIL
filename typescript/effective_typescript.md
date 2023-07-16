- [Materials](#materials)
- [Getting to Know TypeScript](#getting-to-know-typescript)
  - [item 1: Understand the Relationship Between TypeScript and JavaScript](#item-1-understand-the-relationship-between-typescript-and-javascript)
  - [item 2: Know Which TypeScript Options You’re Using](#item-2-know-which-typescript-options-youre-using)
  - [item 3: Understand That Code Generation Is Independent of Types](#item-3-understand-that-code-generation-is-independent-of-types)
  - [item 4: Get Comfortable with Structural Typing](#item-4-get-comfortable-with-structural-typing)
  - [item 5: Limit Use of the any Type](#item-5-limit-use-of-the-any-type)
- [TypeScript’s Type System](#typescripts-type-system)
  - [item 6: Use Your Editor to Interrogate and Explore the Type System](#item-6-use-your-editor-to-interrogate-and-explore-the-type-system)
  - [item 7: Think of Types as Sets of Values](#item-7-think-of-types-as-sets-of-values)
  - [item 8: Know How to Tell Whether a Symbol Is in the Type Space or Value Space](#item-8-know-how-to-tell-whether-a-symbol-is-in-the-type-space-or-value-space)
  - [item 9: Prefer Type Declarations to Type Assertions](#item-9-prefer-type-declarations-to-type-assertions)
  - [item 10: Avoid Object Wrapper Types (String, Number, Boolean, Symbol, BigInt)](#item-10-avoid-object-wrapper-types-string-number-boolean-symbol-bigint)
  - [item 11: Recognize the Limits of Excess Property Checking](#item-11-recognize-the-limits-of-excess-property-checking)
  - [item 12: Apply Types to Entire Function Expressions When Possible](#item-12-apply-types-to-entire-function-expressions-when-possible)
  - [item 13: Know the Differences Between type and interface](#item-13-know-the-differences-between-type-and-interface)
  - [item 14: Use Type Operations and Generics to Avoid Repeating Yourself](#item-14-use-type-operations-and-generics-to-avoid-repeating-yourself)
  - [item 15: Use Index Signatures for Dynamic Data](#item-15-use-index-signatures-for-dynamic-data)
  - [item 16: Prefer Arrays, Tuples, and ArrayLike to number Index Signatures](#item-16-prefer-arrays-tuples-and-arraylike-to-number-index-signatures)
  - [item 17: Use readonly to Avoid Errors Associated with Mutation](#item-17-use-readonly-to-avoid-errors-associated-with-mutation)
  - [item 18: Use Mapped Types to Keep Values in Sync](#item-18-use-mapped-types-to-keep-values-in-sync)
- [Type Inference](#type-inference)
  - [item 19: Avoid Cluttering Your Code with Inferable Types](#item-19-avoid-cluttering-your-code-with-inferable-types)
  - [item 20: Use Different Variables for Different Types](#item-20-use-different-variables-for-different-types)
  - [item 21: Understand Type Widening](#item-21-understand-type-widening)
  - [item 22: Understand Type Narrowing](#item-22-understand-type-narrowing)
  - [item 23: Create Objects All at Once](#item-23-create-objects-all-at-once)
  - [item 24: Be Consistent in Your Use of Aliases](#item-24-be-consistent-in-your-use-of-aliases)
  - [item 25: Use async Functions Instead of Callbacks for Asynchronous Code](#item-25-use-async-functions-instead-of-callbacks-for-asynchronous-code)
  - [item 26: Understand How Context Is Used in Type Inference](#item-26-understand-how-context-is-used-in-type-inference)
  - [item 27: Use Functional Constructs and Libraries to Help Types Flow](#item-27-use-functional-constructs-and-libraries-to-help-types-flow)
- [Type Design](#type-design)
  - [item 28: Prefer Types That Always Represent Valid States](#item-28-prefer-types-that-always-represent-valid-states)
  - [item 29: Be Liberal in What You Accept and Strict in What You Produce](#item-29-be-liberal-in-what-you-accept-and-strict-in-what-you-produce)
  - [item 30: Don’t Repeat Type Information in Documentation](#item-30-dont-repeat-type-information-in-documentation)
  - [item 31: Push Null Values to the Perimeter of Your Types](#item-31-push-null-values-to-the-perimeter-of-your-types)
  - [item 32: Prefer Unions of Interfaces to Interfaces of Unions](#item-32-prefer-unions-of-interfaces-to-interfaces-of-unions)
  - [item 33: Prefer More Precise Alternatives to String Types](#item-33-prefer-more-precise-alternatives-to-string-types)
  - [item 34: Prefer Incomplete Types to Inaccurate Types](#item-34-prefer-incomplete-types-to-inaccurate-types)
  - [item 35: Generate Types from APIs and Specs, Not Data](#item-35-generate-types-from-apis-and-specs-not-data)
  - [item 36: Name Types Using the Language of Your Problem Domain](#item-36-name-types-using-the-language-of-your-problem-domain)
  - [item 37: Consider “Brands” for Nominal Typing](#item-37-consider-brands-for-nominal-typing)
- [Working with any](#working-with-any)
  - [item 38: Use the Narrowest Possible Scope for any Types](#item-38-use-the-narrowest-possible-scope-for-any-types)
  - [item 39: Prefer More Precise Variants of any to Plain any](#item-39-prefer-more-precise-variants-of-any-to-plain-any)
  - [item 40: Hide Unsafe Type Assertions in Well-Typed Functions](#item-40-hide-unsafe-type-assertions-in-well-typed-functions)
  - [item 41: Understand Evolving any](#item-41-understand-evolving-any)
  - [item 42: Use unknown Instead of any for Values with an Unknown Type](#item-42-use-unknown-instead-of-any-for-values-with-an-unknown-type)
  - [item 43: Prefer Type-Safe Approaches to Monkey Patching](#item-43-prefer-type-safe-approaches-to-monkey-patching)
  - [item 44: Track Your Type Coverage to Prevent Regressions in Type Safety](#item-44-track-your-type-coverage-to-prevent-regressions-in-type-safety)
- [Types Declarations and @types](#types-declarations-and-types)
  - [item 45: Put TypeScript and @types in devDependencies](#item-45-put-typescript-and-types-in-devdependencies)
  - [item 46: Understand the Three Versions Involved in Type Declarations](#item-46-understand-the-three-versions-involved-in-type-declarations)
  - [item 47: Export All Types That Appear in Public APIs](#item-47-export-all-types-that-appear-in-public-apis)
  - [item 48: Use TSDoc for API Comments](#item-48-use-tsdoc-for-api-comments)
  - [item 49: Provide a Type for this in Callbacks](#item-49-provide-a-type-for-this-in-callbacks)
  - [item 50: Prefer Conditional Types to Overloaded Declarations](#item-50-prefer-conditional-types-to-overloaded-declarations)
  - [item 51: Mirror Types to Sever Dependencies](#item-51-mirror-types-to-sever-dependencies)
  - [item 52: Be Aware of the Pitfalls of Testing Types](#item-52-be-aware-of-the-pitfalls-of-testing-types)
- [Writing and Running Your Code](#writing-and-running-your-code)
  - [item 53: Prefer ECMAScript Features to TypeScript Features](#item-53-prefer-ecmascript-features-to-typescript-features)
  - [item 54: Know How to Iterate Over Objects](#item-54-know-how-to-iterate-over-objects)
  - [item 55: Understand the DOM hierarchy](#item-55-understand-the-dom-hierarchy)
  - [item 56: Don’t Rely on Private to Hide Information](#item-56-dont-rely-on-private-to-hide-information)
  - [item 57: Use Source Maps to Debug TypeScript](#item-57-use-source-maps-to-debug-typescript)
- [Migrating to TypeScript](#migrating-to-typescript)
  - [item 58: Write Modern JavaScript](#item-58-write-modern-javascript)
  - [item 59: Use @ts-check and JSDoc to Experiment with TypeScript](#item-59-use-ts-check-and-jsdoc-to-experiment-with-typescript)
  - [item 60: Use allowJs to Mix TypeScript and JavaScript](#item-60-use-allowjs-to-mix-typescript-and-javascript)
  - [item 61: Convert Module by Module Up Your Dependency Graph](#item-61-convert-module-by-module-up-your-dependency-graph)
  - [item 62: Don’t Consider Migration Complete Until You Enable noImplicitAny](#item-62-dont-consider-migration-complete-until-you-enable-noimplicitany)

----

# Materials

* [Effective Typescript](https://effectivetypescript.com/)
  * [src](https://github.com/danvk/effective-typescript)

# Getting to Know TypeScript
## item 1: Understand the Relationship Between TypeScript and JavaScript
## item 2: Know Which TypeScript Options You’re Using
## item 3: Understand That Code Generation Is Independent of Types
## item 4: Get Comfortable with Structural Typing
## item 5: Limit Use of the any Type
# TypeScript’s Type System
## item 6: Use Your Editor to Interrogate and Explore the Type System
## item 7: Think of Types as Sets of Values
## item 8: Know How to Tell Whether a Symbol Is in the Type Space or Value Space
## item 9: Prefer Type Declarations to Type Assertions
## item 10: Avoid Object Wrapper Types (String, Number, Boolean, Symbol, BigInt)
## item 11: Recognize the Limits of Excess Property Checking
## item 12: Apply Types to Entire Function Expressions When Possible
## item 13: Know the Differences Between type and interface
## item 14: Use Type Operations and Generics to Avoid Repeating Yourself
## item 15: Use Index Signatures for Dynamic Data
## item 16: Prefer Arrays, Tuples, and ArrayLike to number Index Signatures
## item 17: Use readonly to Avoid Errors Associated with Mutation
## item 18: Use Mapped Types to Keep Values in Sync
# Type Inference
## item 19: Avoid Cluttering Your Code with Inferable Types
## item 20: Use Different Variables for Different Types
## item 21: Understand Type Widening
## item 22: Understand Type Narrowing
## item 23: Create Objects All at Once
## item 24: Be Consistent in Your Use of Aliases
## item 25: Use async Functions Instead of Callbacks for Asynchronous Code
## item 26: Understand How Context Is Used in Type Inference
## item 27: Use Functional Constructs and Libraries to Help Types Flow
# Type Design
## item 28: Prefer Types That Always Represent Valid States
## item 29: Be Liberal in What You Accept and Strict in What You Produce
## item 30: Don’t Repeat Type Information in Documentation
## item 31: Push Null Values to the Perimeter of Your Types
## item 32: Prefer Unions of Interfaces to Interfaces of Unions
## item 33: Prefer More Precise Alternatives to String Types
## item 34: Prefer Incomplete Types to Inaccurate Types
## item 35: Generate Types from APIs and Specs, Not Data
## item 36: Name Types Using the Language of Your Problem Domain
## item 37: Consider “Brands” for Nominal Typing
# Working with any
## item 38: Use the Narrowest Possible Scope for any Types
## item 39: Prefer More Precise Variants of any to Plain any
## item 40: Hide Unsafe Type Assertions in Well-Typed Functions
## item 41: Understand Evolving any
## item 42: Use unknown Instead of any for Values with an Unknown Type
## item 43: Prefer Type-Safe Approaches to Monkey Patching
## item 44: Track Your Type Coverage to Prevent Regressions in Type Safety
# Types Declarations and @types
## item 45: Put TypeScript and @types in devDependencies
## item 46: Understand the Three Versions Involved in Type Declarations
## item 47: Export All Types That Appear in Public APIs
## item 48: Use TSDoc for API Comments
## item 49: Provide a Type for this in Callbacks
## item 50: Prefer Conditional Types to Overloaded Declarations
## item 51: Mirror Types to Sever Dependencies
## item 52: Be Aware of the Pitfalls of Testing Types
# Writing and Running Your Code
## item 53: Prefer ECMAScript Features to TypeScript Features
## item 54: Know How to Iterate Over Objects
## item 55: Understand the DOM hierarchy
## item 56: Don’t Rely on Private to Hide Information
## item 57: Use Source Maps to Debug TypeScript
# Migrating to TypeScript
## item 58: Write Modern JavaScript
## item 59: Use @ts-check and JSDoc to Experiment with TypeScript
## item 60: Use allowJs to Mix TypeScript and JavaScript
## item 61: Convert Module by Module Up Your Dependency Graph
## item 62: Don’t Consider Migration Complete Until You Enable noImplicitAny
