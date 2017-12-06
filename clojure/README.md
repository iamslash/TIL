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

```
brew install leiningen
```

* intelliJ
  * install Cursive plugin


# Usage

## Destructuring

Sequential destructuring

```clj
(let [[f s] [1 2]] f) ;; 1
(let [[f s t] [1 2 3]] [f t]) ;; [1 3]
(let [[f] [1 2]] f) ;; 1
(let [[f s t] [1 2]] t) ;; nil
(let [[f & t] [1 2]] t) ;; (2)
(let [[f & t] [1 2 3]] t) ;; (2 3)
(let [[f & [_ t]] [1 2 3]] [f t]) ;; [1 3]
```

Associative destructuring

```clj
(let [{a-value :a c-value :c} {:a 5 :b 6 :c 7}] a-value) ;; 5
(let [{a-value :a c-value :c} {:a 5 :b 6 :c 7}] c-value) ;; 7
(let [{:keys [a c]} {:a 5 :b 6 :c 7}] c) ;; 7
(let [{:syms [a c]} {'a 5 :b 6 'c 7}] c) ;; 7
(let [{:strs [a c]} {:a 5 :b 6 :c 7 "a" 9}] [a c]) ;; [9 nil]
(let [{:strs [a c] :or {c 42}} {:a 5 :b 6 :c 7 "a" 9}] [a c]) ;; [9 42]
```
## Collections

clojure의 collection은 크게 sequential, associative, counted와 같이
세가지 분류로 구분 할 수 있다. set은 sorted set, hash set으로 map은
sorted map, hash map으로 구성된다.

![](collection-properties-venn.pn)

## Concurrency

future

promise

atom

ref

agent

## Macro

defmacro

macroexpand-1
