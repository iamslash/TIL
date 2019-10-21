- [Materials](#materials)
- [Install](#install)
  - [Install on Windows 10](#install-on-windows-10)
  - [Install on OSX](#install-on-osx)
- [Hello World](#hello-world)
- [DSL (Domain Specific Language)](#dsl-domain-specific-language)
- [Basic](#basic)
  - [Reserved Words](#reserved-words)
  - [Data Types](#data-types)
  - [Collections compared with c++ container](#collections-compared-with-c-container)
  - [Collections](#collections)
  - [Multidimensional Array](#multidimensional-array)
  - [Sort](#sort)
  - [Operators](#operators)
  - [Decision Making](#decision-making)
  - [Loops](#loops)
- [Advance](#advance)
  - [Introspection](#introspection)

----

# Materials

* [Groovy](https://code-maven.com/groovy)
  * Groovy Tutotial for Jenkins Pipeline.
* [groovy  Documentation](http://groovy-lang.org/documentation.html)
  * [Differences with Java](http://groovy-lang.org/differences.html)
* [GROOVY TUTORIAL](http://www.newthinktank.com/2016/04/groovy-tutorial/)
  * 한 페이지로 배우는 groovy
* [Groovy Tutorial @ tutorial point](https://www.tutorialspoint.com/groovy/index.htm)
  * groovy tutorial
* [Groovy for Java developers @ youtube](https://www.youtube.com/watch?v=BXRDTiJfrSE)
  * java 개발자를 위한 groovy

# Install

## Install on Windows 10

```
choco install groovy
```

## Install on OSX

```bash
brew install groovy
```

# Hello World

* a.groovy

```groovy
class A {
    static void main(String[] args){
    // Print to the screen
    println("Hello World");
    }
}
```

```bash
groovy a.groovy
```

# DSL (Domain Specific Language)

groovy 의 top level statement 는 argument 주변의 `()` 와 function call 사이의 `.` 를 생략할 수 있다. 이것을 `command chain` 이라고 한다.

```groovy
// equivalent to: turn(left).then(right)
turn left then right
// equivalent to: take(2.pills).of(chloroquinine).after(6.hours)
take 2.pills of chloroquinine after 6.hours
// equivalent to: paint(wall).with(red, green).and(yellow)
paint wall with red, green and yellow
// with named parameters too
// equivalent to: check(that: margarita).tastes(good)
check that: margarita tastes good
// with closures as parameters
// equivalent to: given({}).when({}).then({})
given { } when { } then { }
// equivalent to: select(all).unique().from(names)
select all unique() from names
// equivalent to: take(3).cookies
// and also this: take(3).getCookies()
take 3 cookies
```

다음은 `EmailDs1` 을 DSL 에서 사용하기 위한 구현이다. [Groovy - DSLS @ tutorialpoint](https://www.tutorialspoint.com/groovy/groovy_dsls.htm)

```groovy
class EmailDsl {  
   String toText 
   String fromText 
   String body 
	
   /** 
   * This method accepts a closure which is essentially the DSL. Delegate the 
   * closure methods to 
   * the DSL class so the calls can be processed 
   */ 
   
   def static make(closure) { 
      EmailDsl emailDsl = new EmailDsl() 
      // any method called in closure will be delegated to the EmailDsl class 
      closure.delegate = emailDsl
      closure() 
   }
   
   /** 
   * Store the parameter as a variable and use it later to output a memo 
   */ 
	
   def to(String toText) { 
      this.toText = toText 
   }
   
   def from(String fromText) { 
      this.fromText = fromText 
   }
   
   def body(String bodyText) { 
      this.body = bodyText 
   } 
}

EmailDsl.make { 
   to "Nirav Assar" 
   from "Barack Obama" 
   body "How are things? We are doing well. Take care" 
}
```

# Basic 

## Reserved Words

TODO

## Data Types

TODO

## Collections compared with c++ container

TODO

## Collections

TODO

## Multidimensional Array

TODO

## Sort

TODO

## Operators

TODO

## Decision Making

TODO

## Loops

TODO

# Advance

## Introspection

* [What's the Groovy equivalent to Python's dir()? @ stackoverflow](https://stackoverflow.com/questions/10882469/whats-the-groovy-equivalent-to-pythons-dir)

----

```groovy
// Introspection, know all the details about classes :
// List all constructors of a class
String.constructors.each{println it}

// List all interfaces implemented by a class
String.interfaces.each{println it}

// List all methods offered by a class
String.methods.each{println it}

// Just list the methods names
String.methods.name

// Get the fields of an object (with their values)
d = new Date()
d.properties.each{ println(it) }
```
