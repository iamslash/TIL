# Materials

* [GROOVY TUTORIAL](http://www.newthinktank.com/2016/04/groovy-tutorial/)
  * 한 페이지로 배우는 groovy
* [Groovy Tutorial @ tutorial point](https://www.tutorialspoint.com/groovy/index.htm)
  * groovy tutorial
* [Groovy for Java developers @ youtube](https://www.youtube.com/watch?v=BXRDTiJfrSE)
  * java 개발자를 위한 groovy

# Install on Windows 10

```
choco install groovy
```

* a.groovy

```groovy
class A {
    static void main(String[] args){
    // Print to the screen
    println("Hello World");
    }
}
```

```
groovy a.groovy
```

# Basic
 
## DSL (Domain Specific Language)

groovy 의 top level statement 는 argument 주변의 `()` 를 생략할 수 있다. `command chains` 특성때문에 argument 주변의 `()` 와 chained call 사이의 `.` 를 생략 할 수 있다. 예를 들어 다음과 같은 표현이 가능하다. 이것 때문에 DSL 에 사용된다.

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