# Abstract

clojure에 대해 정리한다.

# Material

* [reagent](https://reagent-project.github.io/)
  * minimal interface between clojurescript and react
* [reframe](https://github.com/Day8/re-frame)
  * a framework for writing SPAs in clojurescript
* [debux](https://github.com/philoskim/debux)
  * clojure, clojurescript 디버깅 툴
* [클로저 Ring으로 하는 웹개발](https://hatemogi.gitbooks.io/ring/content/%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0.html)
* [clojure in y minutes](https://learnxinyminutes.com/docs/ko-kr/clojure-kr/)
* [초보자를 위한 클로저](https://eunmin.gitbooks.io/clojure-for-beginners/content/)
* [clojure design pattern](http://clojure.or.kr/docs/clojure-and-gof-design-patterns.html)
  * GOF design pattern을 clojure에서 이렇게 쉽게 할 수 있다.
* [4clojure](http://clojure.or.kr/docs/clojure-and-gof-design-patterns.html#mediator)
  * clojure를 문제를 풀면서 학습해보자.
* [clojure examples](https://kimh.github.io/clojure-by-example)
* [clojure for java developer](https://github.com/mbonaci/clojure)
* [brave clojure](https://www.braveclojure.com/clojure-for-the-brave-and-true/)
* [Clojure for Java Programmers Part 2 - Rich Hickey @ youtube](https://www.youtube.com/watch?v=hb3rurFxrZ8)
* [Clojure Koans](https://github.com/functional-koans/clojure-koans)
  * interactive learning clojure
* [datomic tutorial](https://github.com/Datomic/day-of-datomic/tree/master/tutorial)
  * datamic 은 clojure 로 만든 database 이다.
* [datomic doc](http://docs.datomic.com/)
* [datomic intro @ github](https://github.com/philoskim/datomic-intro)

# References

* [cursive userguide](https://cursive-ide.com/userguide/)
  * jetbrain의 cursive plugin
* [repl.it](https://repl.it/language/clojure)
  * clojure repl on the web
* [clojure doc](https://clojuredocs.org/)
* [clojure cheatsheet](https://clojure.org/api/cheatsheet)

# Environment

## Windows10

* leiningen 
  * [leiningen.org](https://leiningen.org/#install) 에서 lein.bat를 d:\local\bin에 다운받고 다음을 실행한다. 

```
lein.bat self-install
```

* intelliJ
  * install Cursive plugin
  
## macosx

* leiningen

```bash
brew install leiningen
```

* intelliJ
  * install Cursive plugin


# Usage

## data types

String

```cojure
"Hello world"
```

Boolean

```clojure
true
false
```

Character

```clojure
\c
\u0045 ;; unicode char 45 E
```

Keywords

```clojure
:foo
:bar
```

Number

```clojure
11N ;; BigInteger
11  ;; long
0.1M ;; BigDecimal
```

Ratio

```clojure
11/7
```

Symbol

```clojure
foo-bar
```

nil

```clojure
nil
```

Regular expressions

```clojure
#"\d"
```

## Packages

:import

```clojure
(:import java.util.List)
(:import [java.util ArrayList HashMap])
(ns a.foo
  (:import [java.util Date])
```

:require

```clojure
(:require [a.b :refere [:all]])
(:require [a.b :as c))
(c/a-function 5)
(:require [a.b :as c :refer [d-funcion]])
```

## Destructuring

Sequential destructuring

```clojure
(let [[f s] [1 2]] f) ;; 1
(let [[f s t] [1 2 3]] [f t]) ;; [1 3]
(let [[f] [1 2]] f) ;; 1
(let [[f s t] [1 2]] t) ;; nil
(let [[f & t] [1 2]] t) ;; (2)
(let [[f & t] [1 2 3]] t) ;; (2 3)
(let [[f & [_ t]] [1 2 3]] [f t]) ;; [1 3]
```

Associative destructuring

```clojure
(let [{a-value :a c-value :c} {:a 5 :b 6 :c 7}] a-value) ;; 5
(let [{a-value :a c-value :c} {:a 5 :b 6 :c 7}] c-value) ;; 7
(let [{:keys [a c]} {:a 5 :b 6 :c 7}] c) ;; 7
(let [{:syms [a c]} {'a 5 :b 6 'c 7}] c) ;; 7
(let [{:strs [a c]} {:a 5 :b 6 :c 7 "a" 9}] [a c]) ;; [9 nil]
(let [{:strs [a c] :or {c 42}} {:a 5 :b 6 :c 7 "a" 9}] [a c]) ;; [9 42]
```
## implementing interfaces and protocols

```clojure
(import '(javax.swing JFrame Jlabel JTextField JButton)
        '(java.awt.event ActionListener)
        '(java.awt GridLayout))
(defn sample []
  (let [frame (JFrame. "Hello")
      sample-button (JButton. "Hello")]
    (.addActionListener
      sample-button
      (reify ActionListener
          (actionPerformed
           [_ evt]
           (println "Hello World"))))
    (doto frame
      (.add sample-button)
      (.setSize 100 40)
      (.setVisible true))))
```


## Collections

clojure의 collection은 크게 sequential, associative, counted와 같이
세가지 분류로 구분 할 수 있다. set은 sorted set, hash set으로 map은
sorted map, hash map으로 구성된다.

![](img/collection-properties-venn.png)

vector

```clojure
[1 2 3 4 5]
```

list

```clojure
(1 2 3 4 5)
```

hash map

```clojure
{:one 1 :two 2}
(hash-map :one 1 :two 2)
```

sorted map

```clojure
(sorted-map :one 1 :two 2) ;; {:one 1, :two 2)
(sorted-map-by > 1 :one 5 :five 3 :three) ;; {5 :five, 3 :three, 1 :one}
```

hash set

```clojure
#{1 2 3 4 5}
```

sorted set

```clojure
(doseq [x (->> (sorted-set :b :c :d)
               (map name))]
    (println x))
;; b
;; c
;; d
```

union, difference, and intersection

```clojure
(def a #{:a :b :c :d :e})
(def b #{:a :d :h :i :j :k})
(require '[clojure.set :as s])
(s/union a b) ;; #{:e :k :c :j :h :b :d :i :a}
(s/difference a b) ;; #{:e :c :b}
(s/intersection a b) ;; #{:d :a}
```

## Polymorphism

multimethod

```clojure
(defn avg [& coll]
  (/ (apply + coll) (count coll)))
(defn get-race [& ages]
  (if (> (apply avg ages) 120)
    :timelord
    :human))
(defmulti travel get-race)
(defmethod travel :timelord [& ages]
  (str (count ages) " timelords travelling by tardis"))
(defmethod travel :human [& ages]
  (str (count ages) " human travelling by car"))
(travel 2000 1000 100 200)
;; "4 timelords travelling by tardis"
(travel 70 20 100 40)
;; "4 humans travelling by car
```

protocol

```clojure
(defprotocol Shape
  "This is a protocol for shapes"
  (perimeter [this] "Calculate the perimeter of this shape")
  (area [this] "Calculate the area of this shape"))
```

record

```clojure
(defrecord Square [side]
  Shape
  (perimeter [{:keys [side]}] (* 4 side))
  (area [{:keys [side]}] (* side side)))
(Square. 5)
```

## Concurrency

identity and state

promise

```clojure

```

future

```clojure

```

STM (software transaction memory) and ref

```clojure

```

atom

```clojure

```

agent

```clojure

```

validator

```clojure

```

watcher

```clojure

```

## Macro

defmacro

```clojure

```

macroexpand-1

```clojure

```

