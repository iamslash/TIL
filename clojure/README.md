- [Abstract](#abstract)
- [Masterials](#masterials)
- [References](#references)
- [Install](#install)
  - [Windows10](#windows10)
  - [osx](#osx)
- [Basic Usages](#basic-usages)
  - [REPL](#repl)
  - [REPL on emacs](#repl-on-emacs)
  - [Compile & Run](#compile--run)
  - [Collections compared c++ container](#collections-compared-c-container)
  - [Collections](#collections)
  - [Data types](#data-types)
  - [Packages](#packages)
  - [Variables](#variables)
  - [Operators](#operators)
  - [Loops](#loops)
  - [Decision Making](#decision-making)
  - [Functions](#functions)
  - [File I/O](#file-io)
  - [Strings](#strings)
  - [Lists](#lists)
  - [Sets](#sets)
  - [Vectors](#vectors)
  - [Maps](#maps)
  - [Namespaces](#namespaces)
  - [Exception Handling](#exception-handling)
  - [Sequences](#sequences)
  - [Regular Expressions](#regular-expressions)
  - [Destructuring](#destructuring)
  - [Date & Time](#date--time)
  - [Atoms](#atoms)
  - [Metadata](#metadata)
  - [StructMaps](#structmaps)
  - [Agents](#agents)
  - [Watchers](#watchers)
  - [Macro](#macro)
  - [Reference Values](#reference-values)
  - [Dtabases](#dtabases)
  - [Java Interface](#java-interface)
  - [Implementing interfaces and protocols](#implementing-interfaces-and-protocols)
  - [Polymorphism](#polymorphism)
  - [Concurrency](#concurrency)

-----------------------------------

# Abstract

clojure에 대해 정리한다.

# Masterials

* [learn clojure in y minutes](https://learnxinyminutes.com/docs/ko-kr/clojure-kr/)
  * 짧은 시간에 ruby 가 어떠한 언어인지 파악해보자
* [clojure @ tutorialspoint](https://www.tutorialspoint.com/clojure/)
* [클로저 시작하기](https://github.com/eunmin/getting-started-clojure/wiki/%ED%81%B4%EB%A1%9C%EC%A0%80-%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0)
  * 실행하기, 컴파일하기
* [Clojure 병행성 @ github](https://github.com/eunmin/clojure-study/wiki/%5B%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%ED%81%B4%EB%A1%9C%EC%A0%80%5D-6%EC%9E%A5-%EB%B3%91%ED%96%89%EC%84%B1)
* [Clojure Programming](www.clojurebook.com/)
  * Clojure의 internal을 가장 잘 설명한 책이다.
* [reagent](https://reagent-project.github.io/)
  * minimal interface between clojurescript and react
* [reframe](https://github.com/Day8/re-frame)
  * a framework for writing SPAs in clojurescript
* [debux](https://github.com/philoskim/debux)
  * clojure, clojurescript 디버깅 툴
* [클로저 Ring으로 하는 웹개발](https://hatemogi.gitbooks.io/ring/content/%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0.html)
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

# Install

## Windows10

* windows 용 command line 도구는 아직 없다. (20181208)
* [leiningen.org](https://leiningen.org/#install) 에서 lein.bat를 d:\local\bin에 다운받는다.

```bash
> lein repl
>> (load-file 'a.clj')
```

## osx

* [Getting Started @ clojure](https://clojure.org/guides/getting_started)

```bash
> brew install clojure
> clj a.clj
```

# Basic Usages

## REPL

```bash
> clj
> lein repl
```

## REPL on emacs

```
M-x cider-jackin-in
```

## Compile & Run

```bash
> clj a.clj
> lein repl
>> (load-file "a.clj")
```


## Collections compared c++ container

| c++                  | clojure              | 
|:---------------------|:---------------------|
| `if, else`           | `if, else`           |
| `for, while`         | `while, doseq, dotimes, loop, recur` |
| `array`              | `PersistentVector`   |
| `vector`             | `PersistentVector`   |
| `deque`              | ``                   |
| `forward_list`       | ``                   |
| `list`               | `PersistentList`     |
| `stack`              | ``                   |
| `queue`              | ``                   |
| `priority_queue`     | ``                   |
| `set`                | `PersistentTreeSet`  |
| `multiset`           | ``                   |
| `map`                | `PersistenTreeMap`   |
| `multimap`           | ``                   |
| `unordered_set`      | `PersistentHashSet`  |
| `unordered_multiset` | ``                   |
| `unordered_map`      | `PersistentArrayMap` |
| `unordered_multimap` | ``                   |

## Collections

clojure 의 collection 은 크게 `sequential, associative, counted` 와 같이
세가지 분류로 구분 할 수 있다. `set` 은 `sorted set, hash set` 으로 `map` 은
`sorted map, hash map` 으로 구성된다.

![](img/collection-properties-venn.png)

* vector (PersistentVector)

```clojure
[1 2 3 4 5]
```

* list (PersistentList)

```clojure
(1 2 3 4 5)
```


* hash set (PersistentHashSet)

```clojure
#{1 2 3 4 5}
```

* sorted set (PersistentTreeSet)

```clojure
(doseq [x (->> (sorted-set :b :c :d)
               (map name))]
    (println x))
;; b
;; c
;; d
```

* union, difference, and intersection

```clojure
(def a #{:a :b :c :d :e})
(def b #{:a :d :h :i :j :k})
(require '[clojure.set :as s])
(s/union a b) ;; #{:e :k :c :j :h :b :d :i :a}
(s/difference a b) ;; #{:e :c :b}
(s/intersection a b) ;; #{:d :a}
```

* hash map (PersistentArrayMap)

```clojure
{:one 1 :two 2}
(hash-map :one 1 :two 2)
```

* sorted map (PersistentTreeMap)

```clojure
(sorted-map :one 1 :two 2) ;; {:one 1, :two 2)
(sorted-map-by > 1 :one 5 :five 3 :three) ;; {5 :five, 3 :three, 1 :one}
```

## Data types

* String

```cojure
"Hello world"
```

* Boolean

```clojure
true
false
```

* Character

```clojure
\c
\u0045 ;; unicode char 45 E
```

* Keywords

```clojure
:foo
:bar
```

* Number

```clojure
1234   ;; Decimal Integers
012    ;; Octal Numbers
0xFF   ;; Hexadecimal Numbers
2r1000 ;; Radix Numbers

12.34    ;; 32bit Floating Point
1.35e-12 ;; 32bit Floating Point Scientific Notation

11N ;; BigInteger
11  ;; long
0.1M ;; BigDecimal
```

* Ratio

```clojure
11/7
```

* Symbol

```clojure
foo-bar
```

* nil

```clojure
nil
```

* Atom

```clojure
(atom 3)
```

* Regular expressions

```clojure
#"\d"
```

## Packages

`:import`

```clojure
(:import java.util.List)
(:import [java.util ArrayList HashMap])
(ns a.foo
  (:import [java.util Date])
```

`:require`

```clojure
(:require [a.b :refere [:all]])
(:require [a.b :as c))
(c/a-function 5)
(:require [a.b :as c :refer [d-funcion]])
```

`use`

```clojure
(use 'clojure.string)
(split "a,b,c" #",")
(use '[clojure.string :only [split]])
(split "a,b,c" #",")
```

## Variables

```clojure
;; Variable Declarations
(def x 1)
;; Printing variables
(println x)
```

## Operators

```clojure
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Arithmetic operators
(+ 1 2)
(- 2 1)
(* 2 2)
(/ 3 2)
(inc 5)
(dec 5)
(max 1 2 3)
(min 1 2 3)
(rem 3 2) ; 1

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Relational operators
(= 2 2)
(not= 3 2)
(< 2 3)
(<= 2 3)
(> 3 2)
(>= 3 2)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Logical operators
(and true false)
(or true false)
(not false)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Bitwise operators
(bit-and 2 3)
(bit-or 2 3)
(bit-xor 2 3)
(bit-not 2 3)
```

## Loops

```clojure
;; while 
(defn Example []
  (def x (atom 1))
  (while (< @x 5)
    (do
      (println @x)
      (swap! x inc))))
(Example)

;; doseq
(defn Example []
  (doseq [n [0 1 2]]
    (println n)))
(Example)

;; dotimes
(defn Example []
  (dotimes [n 5]
    (println n)))
(Example)

;; loop
(defn Example []
  (loop [x 10]
    (when (> x 1)
      (println x)
      (recur (- x 2)))))
(Example)
```

## Decision Making

```clojure
;; if
(if (= 2 2)
  (println "Values are equal")
  (println "Values are not equal"))

;; if-do
(if (= 2 2)
  (do (println "Both the values are equal")
      (println "true"))
  (do (println "Both the values are not equal")
      (println "false")))

;; nested if
(if (and (= 2 2) (= 3 3))
  (println "Values are equal")
  (println "Values are not equal"))

;; case
(def x 5)
(case x 
  5 (println "x is 5")
  6 (println "x is 6")
  (println "x is neither 5 nor 6"))

;; cond
(def x 5)
(cond 
  (= x 5) (println "x is 5")
  (= x 6) (println "x is 6")
  :else (println "x is not defined"))
```

## Functions

```clojure
;; Defining function
(defn Example []
  (def x 1)
  (println x))

;; Anonymous Functions
(defn Example []
  ((fn [x] (* 2 x)) 2))

;; Functions with Multiple Arguments
(defn demo [] (* 2 2))
(defn demo [x] (* 2 x))
(defn demo [x y] (* 2 x y))

;; Variadic Functions
(defn demo [msg & others]
  (str msg (clojure.string/join " " others)))
(demo "Hello" "This" "is" "the" "message")

;; Higher Order Functions
(filter even? (range 0 10))

```

## File I/O

```clojure
;; Reading the contents of a file as an entire string
(defn Example []
  (def s (slurp "a.txt"))
  (println s))

;; Reading the contents of a file one line at a time
(defn Example []
  (with-open [rdr (clojure.java.io/reader "a.txt")]
  (reduce conj [] (line-seq rdr))))

;; Writing to files
(defn Example []
  (with-open [w (clojure.java.io/writer "a.txt" :append true)]
    (.write w (str "foo" "bar"))))

;; Checking to see if a file exists
(defn Example []
  (println (.exists (clojure.java.io/file "a.txt"))))
```

## Strings

```clojure
;; str
(str "Hello" "World")
;; format
(format "Hello, %s" "World")
(format "Pad with leading zeros %06d" 1234)
;; count
(count "Hello")
;; subs
(subs "Hello World" 2 5)
;; compare
(compare "Hello" "hello")
;; lower-case
(clojure.string/lower-case "HelloWorld")
;; upper-case
(clojure.string/upper-case "HelloWorld")
;; join
(clojure.string/jon ", " [1 2 3])
;; split
(clojure.string/split "Hello World" #" ")
;; split-lines
(clojure.string/split-lines "Hello\nWorld")
;; reverse
(reverse "Hello World")
;; replace
(clojure.string/replace "The tutorial is about Groovy" #"Groovy" "Clojure")
;; trim
(clojure.string/trim " White spaces ")
;; triml
(clojure.string/triml " White spaces ")
;; trimr 
(clojure.string/trimr " White spaces ")
```

## Lists

```clojure
;; list*
(list* 1 [2, 3])
;; first
(first (list 1 2 3))
;; nth
(nth (list 1 2 3) 0)
;; (cons elementlst lst)
(cons 0 (list 1 2 3)) ;; (0 1 2 3)
;; (conj lst elementlst)
(conj (list 1 2 3) 4 5) ;; (5 4 1 2 3)
;; (rest lst)
(rest (list 1 2 3))  ;; (2 3)
```

## Sets

```clojure
;; sorted-set
(sorted-set 3 2 1)  ;; #{1,2,3}
;; get
(get (set '(3 2 1)) 2) ;; 2
(get (set '(3 2 1)) 1) ;; 1
;; contains?
(contains? (set '(3 2 1)) 2) ;; true
;; (conj setofelements x)
(conj (set '(3 2 1)) 5)
;; (disj setofelements x)
(disj (set '(3 2 1)) 2)
;; (union set1 set2)
(clojure.set/union #{1 2} ${3 4})  ;; #{1 4 3 2}
;; difference
(clojure.set/difference #{1 2} #{2 3})
;; intersection
(clojure.set/intersection #{1 2} #{2 3})
;; subset?
(clojure.set/subset? #{1 2} #{2 3})
;; superset?
(clojure.set/superset? #{1 2} #{2 3})
```

## Vectors

```clojure
;; (vector-of t setofelements)
(vector-of :nint 1 2 3)
;; (nth vec index)
(nth (vector 1 2 3) 0)
;; (get vec index)
(get (vector 3 2 1) 2) ;; 2
;; (conj vec x)
(conj (vector 3 2 1) 5)
;; (pop vec)
(pop (vector 3 2 1)) ;; [3 2]
;; (subvec vec start end)
(subvec (ve tor 1 2 3 4 5 6 7) 2 5)
```

## Maps

```clojure
;; (get hmap key)
(get (hash-map "z" "1" "b" "2" "a" "3") "b") ;; 2
;; (contains? hmap key)
(contains? (hash-map "z" "1" "b" "2" "a" "3") "b")
;; find
(find (hash-map "z" "1" "b" "2" "a" "3") "b") ;; [b 2]
;; keys
(keys (hash-map "z" "1" "b" "2" "a" "3")) ;; (z a b)
;; vals
(vals (hash-map "z" "1" "b" "2" "a" "3")) ;; (1 2 3)
;; (dissoc hmap key)
(dissoc (hash-map "z" "1" "b" "2" "a" "3") "b") ;; {z 1, a 3}
;; (merge hmap1 hmap2)
(def hm1 (hash-map "z" 1 "b" 2 "a" 3))
(def hm2 (hash-map "x" 2 "h" 5 "i" 7))
(merge hm1 hm2)
;; (merge-with f hmap1 hmap2)
(def hm1 (hash-map "z" 1 "b" 2 "a" 3))
(def hm2 (hash-map "a" 2 "h" 5 "i" 7))
(merge-with + hm1 hm2) ;; {z 1, a 5, i 7, b 2, h 5}
;; (select-keys hmap keys)
(select-keys (hash-map "z" 1 "b" 2 "a" 3) ["z" "a"]) ;; {z 1, a 3}
;; (rename-keys hmap keys)
(clojure.set/rename-keys (hash-map "z" 1 "b" 2 "a" 3)
  {"z" "newz" "b" "newb" "a" "newa"}) ;; {newa 3, newb 2, newz 1}
;; (map-invert hmap)
(clojure.set/map-invert (hash-map "z" 1 "b" 2 "a" 3)) ;; {1 z, 3 a, 2 b}
```

## Namespaces

```clojure
;; *ns* current namespace
(println *ns*)
;; (ns namespace-name)
(ns iamslash
  (:require [clojure.set :as set])
  (:gen-class))
;; (alias aliasname namespace-name)
(alias 'string 'clojure.examples.hello)
;; all-ns
(println all-ns)
;; (find-ns namespace-name)
(find-ns 'clojure.string)
;; ns-name returns the name of a particular namespace
(ns-name 'clojure.string)
;; ns-aliases, returns the aliases, which are associated with any namespaces
(ns-aliases 'clojure.core)
;; ns-map, returns a map of all the mappings for the namespace
(ns-map 'clojure.string)
;; (un-alias namespace-name aliasname)
(ns-unalias 'clojure.core 'string)
```

## Exception Handling

```clojure
;; Catching Exceptions
; (try
;    (//Protected code)
;    catch Exception e1)
; (//Catch block)
(try 
  (def s (slurp "a.txt"))
  (println s)
  (catch Exception e 
    (println (str "caught exception: " (.getMessage e)))))
;; Multiple Catch Blocks
(try
  (def s (slurp "a.txt"))
  (println s)
  (catch java.io.FileNotFoundException e 
    (println (str "caught file exception: " (.getMessage e))))
  (catch Exception e 
    (println (str "caught exception: " (.getMessage e))))
  (println "Let's move on"))
;; finally
(try
  (def s (slurp "a.txt"))
  (println s)
  (catch java.io.FileNotFoundException e 
    (println (str "caught file exception: " (.getMessage e))))
  (catch Exception e 
    (println (str "caught exception: " (.getMessage e))))
  (finally 
    (println "This is out final block"))
  (println "Let's move on"))
```

## Sequences

```clojure
;; (cons x seq)
(cons 0 (seq [1 2 3])) ;; (0 1 2 3)
;; (conj seq x)
(conj [1 2 3] 4) ;; (1 2 3 4)
;; (concat seq1 seq2)
(concat (seq [1 2] (seq [3 4]))) ; (1 2 3 4)
;; distinct
(distinct (seq [1 1 2 2]))
;; reverse
(reverse (seq [1 2 3]))
;; first
(first (seq [1 2 3]))
;; last
(last (seq [1 2 3]))
;; rest
(rest (seq [1 2 3]))
;; sort
(sort (seq [3 2 1]))
;; (drop num seq1)
(drop 2 (seq [5 4 3 2 1])) ; (3 2 1)
;; (take-last num seq1)
(take-last 2 (seq [5 4 3 2 1])) ; (2 1)
;; (take num seq1)
(take 2 (seq [5 4 3 2 1]))
;; (split-at num seq1)
(split-at 2 (seq [5 4 3 2 1])) ; [(5 4) (3 2 1)]
```

## Regular Expressions

```clojure
;; (re-pattern pat)
(re-pattern "\\d+")
;; (re-find pat str)
(re-find (re-pattern "\\d+") "abc123de") ; 123
;; replace
(clojure.string/replace "abc123de" 
  (re-pattern "\\d+" "789")) ; abc789de
;; replace-first
(defn Example []
   (def pat (re-pattern "\\d+"))
   (def newstr1 (clojure.string/replace "abc123de123" pat "789"))
   (def newstr2 (clojure.string/replace-first "abc123de123" pat "789"))
   (println newstr1)
   (println newstr2))
;abc789de789
;abc789de123
```

## Destructuring

* Sequential destructuring

`f & t` 에서 `&` 다음의 argument `t` 는 list 이다.

```clojure
(let [[f s] [1 2]] f) ;; 1
(let [[f s t] [1 2 3]] [f t]) ;; [1 3]
(let [[f] [1 2]] f) ;; 1
(let [[f s t] [1 2]] t) ;; nil
(let [[f & t] [1 2]] t) ;; (2)
(let [[f & t] [1 2 3]] t) ;; (2 3)
(let [[f & [_ t]] [1 2 3]] [f t]) ;; [1 3]
```

* Associative destructuring

```clojure
(let [{a-value :a c-value :c} {:a 5 :b 6 :c 7}] a-value) ;; 5
(let [{a-value :a c-value :c} {:a 5 :b 6 :c 7}] c-value) ;; 7
(let [{:keys [a c]} {:a 5 :b 6 :c 7}] c) ;; 7
(let [{:syms [a c]} {'a 5 :b 6 'c 7}] c) ;; 7
(let [{:strs [a c]} {:a 5 :b 6 :c 7 "a" 9}] [a c]) ;; [9 nil]
(let [{:strs [a c] :or {c 42}} {:a 5 :b 6 :c 7 "a" 9}] [a c]) ;; [9 42]
```

## Date & Time

```clojure
;; java.util.Date.
(def date (.toString (java.util.Date.)))
;Tue Mar 01 06:11:17 UTC 2016
;; java.text.SimpleDateFormat
(def date (.format (java.text.SimpleDateFormat. "MM/dd/yyyy") (new java.util.Date)))
; 03/01/2016
;; getTime
(def date (.getTime (java.util.Date.)))
;1456812778160
```

## Atoms

clojure 가 제공하는 mutable data type 중 하나이다. 이름처럼 원자성을 보장한다. 즉 동기화가 보장된다. mutable data type 은 atom, agent 등이 있다.

```clojure
;; (reset! atom-name newvalue)
(def myatom (atom 1))
(println @myatom) ; 1
(reset! myatom 2)
(println @myatom) ; 2
;; (compare-and-set! atom-name oldvalue newvalue)
   (def myatom (atom 1))
   (println @myatom) ; 1
   
   (compare-and-set! myatom 0 3)
   (println @myatom) ; 1
  
   (compare-and-set! myatom 1 3)
   (println @myatom) ; 3
;; (swap! atom-name function)
(def myatom (atom 1))
(println @myatom) ; 1
(swap! myatom inc)
(println @myatom) ; 2
```

## Metadata

부가정보에 해당한다.

```clojure
;; (with-meta obj mapentry)
(defn Example []
  (def my-map (with-meta [1 2 3] {:prop "values"})
  (println (meta my-map))))
(Example) ; {:prop values}
;; (vary-meta obj new-meta)
;; Returns an object of the same type and value as the original object, but with a combined metadata.
(defn Example []
  (def my-map (with-meta [1 2 3] {:prop "values"}))
  (println (meta my-map))
  (def ne-map (vary-meta my-map assoc :newprop "new values"))
  (println (meta ne-map)))
(Example) 
;{:prop values}
;{:prop values, :newprop new values}
```

## StructMaps

```clojure
;; (defstruct structname keys)
(defstruct Employee :EmployeeName :Employeeid)
;; struct
(struct Employee "John" 1)
;; (struct-map structname keyn valuen …. )
(struct-map Employee :EmployeeName "John" :Employeeid 1)
;; Accesing Indivisual Fields
;; :key structure-name
(def emp (struct-map Employee :EmployeeName "John" :Employeeid 1))
(:Employeeid emp)
;; Immutable Nature
(defn Example []
   (defstruct Employee :EmployeeName :Employeeid)
   (def emp (struct-map Employee :EmployeeName "John" :Employeeid 1))
   (println (:EmployeeName emp))
   
   (assoc emp :EmployeeName "Mark")
   (println (:EmployeeName emp)))
(Example) 
;John
;John
(defn Example []
   (defstruct Employee :EmployeeName :Employeeid)
   (def emp (struct-map Employee :EmployeeName "John" :Employeeid 1))
   (def newemp (assoc emp :EmployeeName "Mark"))
   (println newemp))
(Example)
;{:EmployeeName Mark, :Employeeid 1}
;; Adding a New key to the Structure
(defn Example []
   (defstruct Employee :EmployeeName :Employeeid)
   (def emp (struct-map Employee :EmployeeName "John" :Employeeid 1))
   (def newemp (assoc emp :EmployeeRank "A"))
   (println newemp))
(Example)
;{:EmployeeName John, :Employeeid 1, :EmployeeRank A}
```

## Agents

ref, atom과 비슷하게 동기화를 보장하는 reference type 이다. thread pool의 가용 thread에서
별도로 동작한다. 일을 시키고 잃어버리고 싶을때 사용한다. 즉 비동기
태스크에 적당하다.

```clojure
;; (agent state)
(defn Example []
  (def counter (agent 0))
  (println counter)
  (println @counter))
(Example)
;#object[clojure.lang.Agent 0x371c02e5 {:status :ready, :val 0}]
; 0
;; (send agentname function value)
(defn Example []
   (def counter (agent 0))
   (println @counter) 
   (send counter + 100)
   (println "Incrementing Counter")
   (println @counter))
(Example)
;0
;Incrementing Counter
;0

;; (shutdown-agents)
(defn Example []
   (def counter (agent 0))
   (println @counter)
   
   (send counter + 100)
   (println "Incrementing Counter")
   (println @counter)
   (shutdown-agents))
(Example)
;0
;Incrementing Counter
;0

;; send-off
(ns clojure.examples.example
   (:gen-class))
(defn Example []
   (def counter (agent 0))
   (println @counter)   
   (send-off counter + 100)
   (println @counter)
   (println @counter))
(Example)
;0
;0
;0

;; (await-for time agentname)
(defn Example []
   (def counter (agent 0))
   (println @counter)
   
   (send-off counter + 100)
   (println (await-for 100 counter))
   (println @counter)
   
   (shutdown-agents))
(Example)
;0
;true
;100

;; (await agentname)
(defn Example []
   (def counter (agent 0))
   (println @counter)
   
   (send-off counter + 100)
   (await counter)
   (println @counter)
   
   (shutdown-agents))
(Example)
;0
;100

;; agent-error
(defn Example []
   (def my-date (agent (java.util.Date.)))
   (send my-date + 100)
   (await-for 100 my-date)
   (println (agent-error my-date)))
(Example)
;
```

## Watchers

```clojure
;; (add-watch variable :watcher
;;   (fn [key variable-type old-state new-state]))
(defn Example []
   (def x (atom 0))
   (add-watch x :watcher
      (fn [key atom old-state new-state]
      (println "The value of the atom has been changed")
      (println "old-state" old-state)
      (println "new-state" new-state)))
(reset! x 2))
(Example)
;The value of the atom has been changed
;old-state 0
;new-state 2

;; (remove-watch variable watchname)
(defn Example []
   (def x (atom 0))
   (add-watch x :watcher
      (fn [key atom old-state new-state]
         (println "The value of the atom has been changed")
         (println "old-state" old-state)
         (println "new-state" new-state)))
   (reset! x 2)
   (remove-watch x :watcher)
(reset! x 4))
(Example)
;The value of the atom has been changed
;old-state 0
;new-state 2
```

## Macro

in-line 코드를 생성하기 위해 사용한다.

```clojure
;; (defmacro name [params*] body)
(def a 150)
(defmacro my-if [tset positive negative]
  '(if test positive negative))
(my-if (> a 200)
  (println "Bigger than 200")
  (println "Smaller than 200"))

;; (macroexpand macroname)
(def a 150)
(defmacro my-if [test positive negative]
  '(if test positive negative))
(macroexpand-1
  '(my-if (> a 200)
     (println "Bigger than 200")
     (println "Smaller than 200")))
; (if test positive negative)

;; Macro with arguments
(defn Example []
   (defmacro Simple [arg]
      (list 2 arg))
   (println (macroexpand '(Simple 2))))
(Example)
```

## Reference Values

clojure 가 제공하는 mutable data type 중 하나이다. mutable data type 은 atom, agent, ref 등이 있다.

```clojure
;; (ref x options)
(defn Example []
   (def my-ref (ref 1 :validator pos?))
   (println @my-ref))
(Example)
;1

;; (ref-set refname newvalue)
(defn Example []
   (def my-ref (ref 1 :validator pos?))
   (dosync
      (ref-set my-ref 2))
   (println @my-ref))
(Example)
;2

;; (alter refname fun)
(defn Example []
   (def names (ref []))
   
   (defn change [newname]
      (dosync
         (alter names conj newname)))
   (change "John")
   (change "Mark")
   (println @names))
(Example)
;[John Mark]

;; (dosync expression)
;; Runs the expression (in an implicit do) in a transaction that encompasses expression and any nested calls. Starts a transaction if none is already running on this thread. Any uncaught exception will abort the transaction and flow out of dosync.

;; (commute refname fun)
;; Commute is also used to change the value of a reference type just like alter and ref-set. The only difference is that this also needs to be placed inside a ‘dosync’ block. However, this can be used whenever there is no need to know which calling process actually changed the value of the reference type. Commute also uses a function to change the value of the reference variable.
(defn Example []
   (def counter (ref 0))
   
   (defn change [counter]
      (dosync
         (commute counter inc)))
   (change counter)
   (println @counter)
   
   (change counter)
   (println @counter))
(Example)
;1
;2
```

## Dtabases

```clojure
; (def connection_name {
;    :subprotocol “protocol_name”
;    :subname “Location of mysql DB”
;    :user “username” :password “password” })
(ns A.core
   (:require [clojure.java.jdbc :as sql]))
(defn -main []
   (def mysql-db {
      :subprotocol "mysql"
      :subname "//127.0.0.1:3306/information_schema"
      :user "root"
      :password "shakinstev"})
   (println (sql/query mysql-db
      ["select table_name from tables"]
      :row-fn :table_name)))

;; query select
; clojure.java.jdbc/query dbconn
; ["query"]
;    :row-fn :sequence
(ns A.core
   (:require [clojure.java.jdbc :as sql]))
(defn -main []
   (def mysql-db {
      :subprotocol "mysql"
      :subname "//127.0.0.1:3306/testdb"
      :user "root"
      :password "shakinstev"})
   (println (sql/query mysql-db
      ["select first_name from employee"]
      :row-fn :first_name)))

;; query insert
; clojure.java.jdbc/insert!
;    :table_name {:column_namen columnvalue}
(ns test.core
   (:require [clojure.java.jdbc :as sql]))
(defn -main []
   (def mysql-db {
      :subprotocol "mysql"
      :subname "//127.0.0.1:3306/testdb"
      :user "root"
      :password "shakinstev"})
   (sql/insert! mysql-db
      :employee {:first_name "John" :last_name "Mark" :sex "M" :age 30 :income 30}))

;; query delete
; clojure.java.jdbc/delete!
;    :table_name [condition]
(ns A.core
   (:require [clojure.java.jdbc :as sql]))
(defn -main []
   (def mysql-db {
      :subprotocol "mysql"
      :subname "//127.0.0.1:3306/testdb"
      :user "root"
      :password "shakinstev"})
   (println (sql/delete! mysql-db
      :employee ["age = ? " 30])))

;; query update
; clojure.java.jdbc/update!
;    :table_name
; {setcondition}
; [condition]
(ns test.core
   (:require [clojure.java.jdbc :as sql]))
(defn -main []
   (def mysql-db {
      :subprotocol "mysql"
      :subname "//127.0.0.1:3306/testdb"
      :user "root"
      :password "shakinstev"})
   (println (sql/update! mysql-db
      :employee
      {:income 40}
      ["age = ? " 30])))

;; transaction
(ns A.core
   (:require [clojure.java.jdbc :as sql]))
(defn -main []
   (def mysql-db {
      :subprotocol "mysql"
      :subname "//127.0.0.1:3306/testdb"
      :user "root"
      :password "shakinstev"})
   (sql/with-db-transaction [t-con mysql-db]
      (sql/update! t-con
         :employee
         {:income 40}
         ["age = ? " 30])))
```

## Java Interface

Java Object 는 `(Date.)` 과 같이 `.` 을 클래스다음에 추가하여 생성한다. 

```clojure
;; calling java methods
(ns Project
   (:gen-class))
(defn Example []
   (println (.toUpperCase "Hello World")))
(Example)
; HELLO WORLD

;; calling java methods with params
(ns Project
   (:gen-class))
(defn Example []
   (println (.indexOf "Hello World","e")))
(Example)
;1

;; creating Java Objects
(ns Project
   (:gen-class))
(defn Example []
   (def str1 (new String "Hello"))
   (println str1))
(Example)
; Hello
(ns Project
   (:gen-class))
(defn Example []
   (def my-int(new Integer 1))
   (println (+ 2 my-int)))
(Example)
; 3

;; import
(ns Project
   (:gen-class))
(import java.util.Stack)
(defn Example []
   (let [stack (Stack.)]
   (.push stack "First Element")
   (.push stack "Second Element")
   (println (first stack))))
(Example)
;First Element

;; Java Built-in functions
(ns Project
   (:gen-class))
(defn Example []
   (println (. Math PI)))
(Example)
;3.141592653589793
(ns Project
   (:gen-class))
(defn Example []
   (println (.. System getProperties (get "java.version"))))
; (. System getProperty "java.vm.version")
; (System/getProperty "java.vm.version")
(Example)
;1.8.0_45
```

## Implementing interfaces and protocols

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

## Polymorphism

* multimethod

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

* protocol

```clojure
(defprotocol Shape
  "This is a protocol for shapes"
  (perimeter [this] "Calculate the perimeter of this shape")
  (area [this] "Calculate the area of this shape"))
```

* record

```clojure
(defrecord Square [side]
  Shape
  (perimeter [{:keys [side]}] (* 4 side))
  (area [{:keys [side]}] (* side side)))
(Square. 5)
```

## Concurrency

* identity and state

identity는 state를 가리키고 있다. state는 항상 변할 수 있다. state가
변할때 마다 그때 그때의 state는 별도의 존재들이다.

* promise

이미 값이 채워졌다면 역참조할때 바로 값을 얻어올 수 있다. 그러나 값이
채워져있지 않다면 값이 채워질때까지 블록되어 기다린다. promise는
캐싱된다. 값은 dliver를 이용하여 채운다.

```clojure
(in-ns 'clojure-concurrency.core)
(def p (promise))
(start-thread
  #(do
    (deref p)
    (println "Hello World")))
(deliver p 5)
```

* future

promise와 비슷하지만 항상 다른 thread에서 동작한다는 큰 차이가 있다.
thread pool에서 가용 thread하나를 선택하여 실행한다. 만약 가용
thread가 없다면 기다릴 수 있기 때문에 주의가 필요하다.

```clojure
(def f (future 
          (Thread/sleep 20000) "Hello World"))
(println @f)
```

* STM (software transaction memory) and ref

ref-set을 이용해 ref를 변경 할 수 있다. 이때 dosync를 반드시 이용하여
transaction을 만들어 내야 한다. 그렇지 않으면
java.lang.IllegalStateException 가 발생한다.

STM은 DATABASE의 transaction과 비슷하게 다음과 같은 특성을 갖는다.

* atomicity(원자성) : 하나의 트랜잭션에서 여러개의 ref를 갱신하는 경우 동시에 갱신되는 것이 보장된다.
* consistency(일관성) : ref는 validation function(유효성 확인 함수)를 가질 수 있는데 이 함수중 하나가 실패해도 rollback된다.
* isolation(고립성) : 실행중인 트랜잭션의 부분적 내용은 다른 트랜잭션에서 접근 할 수 없다.
* no durability(영구성) : DB는 disk에 저장가능하지만 clojure는 그렇지 않다.

```clojure
(def account (ref 20000))
(dosync (ref-set account 10))
(deref account)

(defn test []
  (dotimes [n 5]
    (println n @account)
    (Thread/sleep 2000))
  (ref-set account 90))
  
(future (dosync (test)))
(Thread/sleep 1000)
(dosync (ref-set account 5))
```

* atom

ref와 비슷하게 동기화를 보장하지만 transaction은 발생하지 않기 때문에
훨씬 가볍다.

```clojure
(def a (atom "Foo, Bar, Baz"))
(deref a)
```

* agent

ref, atom과 비슷하게 동기화를 보장하지만 thread pool의 가용 thread에서
별도로 동작한다. 일을 시키고 잃어버리고 싶을때 사용한다. 즉 비동기
태스크에 적당하다.

```clojure
(def counter (agent 0))
(send counter inc)
```
