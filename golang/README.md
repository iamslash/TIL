# Abstract

golang에 대해 정리한다. IDE는 VScode가 좋다.

# Materials

* [1ambda golang](https://github.com/1ambda/golang)
  * 유용한 go links
* [effective go](https://golang.org/doc/effective_go.html)  

## Language

* [golang cheatsheet](https://github.com/a8m/go-lang-cheat-sheet)
  * 최고의 요약
* [A Tour of Go video](https://research.swtch.com/gotour)
  * interface, concurrency 에 관한 screencast

## Tools

* [Go Tooling in Action](https://www.youtube.com/watch?v=uBjoTxosSys)
  * go run, go build, go install, go test, go list, go doc, go-wrk,
    go-torch, debugging등등을 다루는 킹왕짱 동영상

## Best Practices

* [Go best practices, six years in @ infoq](https://www.infoq.com/presentations/go-patterns)
  * [article](https://peter.bourgon.org/go-best-practices-2016/)
* [Twelve Go Best Practices @ youtube](https://www.youtube.com/watch?v=8D3Vmm1BGoY)
  * [slide](https://talks.golang.org/2013/bestpractices.slide#1)
* [Go Proverbs - Rob Pike](https://go-proverbs.github.io/)
  * rob pike가 전하는 golang철학

## Concurrency

## Error Handling

## Interface

## Struct

## Pointer

## Map, Slice

## Logging

## Encoding, JSON
  
# References

* [golang doc](https://golang.org/doc/)

# Language

## 

# Tools

## go

주로 사용하는 command는 다음과 같다. 도움말은 go help를 이용하자.

```
go run
go build
go install
go get
go fmt
go vet
```

## go-wrk

an HTTP benchmarking tool

```
go-wrk -c 5 -d 5 http://localhost:8080/
```

## go-torch

Tool for stochastically profiling Go programs. Collects stack traces
and synthesizes them into a flame graph. Uses Go's built in pprof
library.

# Debug

VS Code를 사용한다면 debug mode로 launch하자.

# Test

tests from VS Code

code coverage

table driven tests

# Benchmarks

```
go test-bench
```

# Profile

go-torch

# Examples

## Building a simple web server

net/http, errcheck

## regexp

regexp
