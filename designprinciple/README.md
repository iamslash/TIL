- [Abstract](#abstract)
- [Materials](#materials)
- [KISS](#kiss)
- [YAGNI](#yagni)
- [Do The Simplest Thing That Could Possibly Work](#do-the-simplest-thing-that-could-possibly-work)
- [Separation of Concerns](#separation-of-concerns)
- [Keep things DRY](#keep-things-dry)
- [Code For The Maintainer](#code-for-the-maintainer)
- [Avoid Premature Optimization](#avoid-premature-optimization)
- [Minimise Coupling](#minimise-coupling)
- [Law of Demeter](#law-of-demeter)
- [Composition Over Inheritance](#composition-over-inheritance)
- [Orthogonality](#orthogonality)
- [Robustness Principle](#robustness-principle)
- [Inversion of Control](#inversion-of-control)
- [Maximise Cohesion](#maximise-cohesion)
- [Liskov Substitution Principle](#liskov-substitution-principle)
- [Open/Closed Principle](#openclosed-principle)
- [Single Responsibility Principle](#single-responsibility-principle)
- [Hide Implementation Details](#hide-implementation-details)
- [Curly's Law](#curlys-law)
- [Encapsulate What Changes](#encapsulate-what-changes)
- [Interface Segregation Principle](#interface-segregation-principle)
- [Boy-Scout Rule](#boy-scout-rule)
- [Command Query Separation](#command-query-separation)
- [Murphy's Law](#murphys-law)
- [Brooks's Law](#brookss-law)
- [Linus's Law](#linuss-law)

----

# Abstract

Software Deisgn Principle 에 대해 정리한다.

# Materials

* [Programming Principles](https://java-design-patterns.com/principles/)

# KISS

* [Keep It Simple Stupid (KISS)](https://java-design-patterns.com/principles/#kiss)

# YAGNI

YAGNI stands for "you aren't gonna need it": don't implement something until it is necessary.

* [YAGNI](https://java-design-patterns.com/principles/#yagni)

# Do The Simplest Thing That Could Possibly Work

[Do The Simplest Thing That Could Possibly Work](https://java-design-patterns.com/principles/#do-the-simplest-thing-that-could-possibly-work)

# Separation of Concerns

Separation of concerns is a design principle for separating a computer program
into distinct sections, such that each section addresses a separate concern.

* [Separation of Concerns](https://java-design-patterns.com/principles/#separation-of-concerns)

# Keep things DRY

Every piece of knowledge must have a single, unambiguous, authoritative
representation within a system.

* [Keep things DRY](https://java-design-patterns.com/principles/#keep-things-dry)

# Code For The Maintainer

comment 잘해라???

* [Code For The Maintainer](https://java-design-patterns.com/principles/#code-for-the-maintainer)

# Avoid Premature Optimization

필요이상으로 최적화지 말아라???

* [Avoid Premature Optimization](https://java-design-patterns.com/principles/#avoid-premature-optimization)

# Minimise Coupling

Coupling between modules/components is their degree of mutual interdependence;
lower coupling is better. In other words, coupling is the probability that code
unit "B" will "break" after an unknown change to code unit "A".

* [Minimise Coupling](https://java-design-patterns.com/principles/#minimise-coupling)

# Law of Demeter

Don't talk to strangers.

* [Law of Demeter](https://java-design-patterns.com/principles/#law-of-demeter)

# Composition Over Inheritance

상속보다는 컴포지션이 좋다. LSP 가 깨진다. "has-a" 관계이면 컴포지션을 "is-a"
관계이면 상속을 해라.

* [Composition Over Inheritance](https://java-design-patterns.com/principles/#composition-over-inheritance)

# Orthogonality

Things that are not related conceptually should not be related in the system.

* [Orthogonality](https://java-design-patterns.com/principles/#orthogonality)

# Robustness Principle

Be conservative in what you do, be liberal in what you accept from others

* [Robustness Principle](https://java-design-patterns.com/principles/#robustness-principle)

# Inversion of Control

Inversion of Control is also known as the Hollywood Principle, "Don't call us,
we'll call you". It is a design principle in which custom-written portions of a
computer program receive the flow of control from a generic framework. Inversion
of control carries the strong connotation that the reusable code and the
problem-specific code are developed independently even though they operate
together in an application.

* [Inversion of Control](https://java-design-patterns.com/principles/#inversion-of-control)

# Maximise Cohesion

Cohesion of a single module/component is the degree to which its
responsibilities form a meaningful unit; higher cohesion is better.

* [Maximise Cohesion](https://java-design-patterns.com/principles/#maximise-cohesion)

# Liskov Substitution Principle

Objects in a program should be replaceable with instances of their subtypes
without altering the correctness of that program.

* [Liskov Substitution Principle](https://java-design-patterns.com/principles/#liskov-substitution-principle)

# Open/Closed Principle

Software entities (e.g. classes) should be **open for extension**, but **closed
for modification**. I.e. such an entity can allow its behavior to be modified
without altering its source code.

* [Open/Closed Principle](https://java-design-patterns.com/principles/#openclosed-principle)

# Single Responsibility Principle

A class should never have more than one reason to change.

* [Single Responsibility Principle](https://java-design-patterns.com/principles/#single-responsibility-principle)

# Hide Implementation Details

A software module hides information (i.e. implementation details) by providing
an interface, and not leak any unnecessary information.

* [Hide Implementation Details](https://java-design-patterns.com/principles/#hide-implementation-details)

# Curly's Law

Curly's Law is about choosing a single, clearly defined goal for any particular
bit of code: Do One Thing.

* [Curly's Law](https://java-design-patterns.com/principles/#curlys-law)

# Encapsulate What Changes

A good design identifies the hotspots that are most likely to change and
encapsulates them behind an API. When an anticipated change then occurs, the
modifications are kept local.

* [Encapsulate What Changes](https://java-design-patterns.com/principles/#encapsulate-what-changes)

# Interface Segregation Principle

interface 뚱둥하면 잘게 쪼개라.

* [Interface Segregation Principle](https://java-design-patterns.com/principles/#interface-segregation-principle)
  
# Boy-Scout Rule

code 를 처음 봤을 때 보다 깨끗하게 하라. 기회가 있을 때 마다 refactoring 해라.

* [Boy-Scout Rule](https://java-design-patterns.com/principles/#boy-scout-rule)
  
# Command Query Separation

The Command Query Separation principle states that each method should be either
a command that performs an action or a query that returns data to the caller but
not both. Asking a question should not modify the answer.

* [Command Query Separation](https://java-design-patterns.com/principles/#command-query-separation)
  
# Murphy's Law

Anything that can go wrong will go wrong.

* [Murphy's Law](https://java-design-patterns.com/principles/#murphys-law)
  
# Brooks's Law

Adding manpower to a late software project makes it later. 개발자를 추가한다고
바로 생산성이 향상되지는 않는다. Fred Brooks in his famous book 'The Mythical
Man-Month'.

* [Brooks's Law](https://java-design-patterns.com/principles/#brookss-law)
  
# Linus's Law

Given enough eyeballs, all bugs are shallow. code review 할 수 있는 개발자가
많으면 버그는 적어진다. The book 'The Cathedral and the Bazaar' by Eric S.
Raymond.

* [Linus's Law](https://java-design-patterns.com/principles/#linuss-law)
  