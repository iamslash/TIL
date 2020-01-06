- [Abstract](#abstract)
- [Materials](#materials)
- [Go Examples](#go-examples)
  - [Hello binary](#hello-binary)
  - [Hello binary with library](#hello-binary-with-library)
  - [gazelle](#gazelle)
  - [go get](#go-get)
  - [protobuf](#protobuf)
  - [grpc](#grpc)
  - [go docker image](#go-docker-image)

----

# Abstract

Bazel is a build, test application.

# Materials

* [getting started @ bazel](https://docs.bazel.build/versions/master/getting-started.html)
  * [src](https://github.com/bazelbuild/examples)
* [Building Go Services With Bazel @ youtube](https://www.youtube.com/watch?v=v7EAdff-YXQ) 
  * bazel with go 
* [si-you/bazel-golang-examples @ github](https://github.com/si-you/bazel-golang-examples)

# Go Examples

## Hello binary

```bash
$ bazel build //main:hello
```

```
.
├── WORKSPACE
└── main
    ├── BUILD
    └── hello.go
```

* WORKSPACE

```py
workspace(name = "hello")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "842ec0e6b4fbfdd3de6150b61af92901eeb73681fd4d185746644c338f51d4c0",
    urls = [
        "https://storage.googleapis.com/bazel-mirror/github.com/bazelbuild/rules_go/releases/download/v0.20.1/rules_go-v0.20.1.tar.gz",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.20.1/rules_go-v0.20.1.tar.gz",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains()
```

* BUILD

```py
load("@io_bazel_rules_go//go:def.bzl", "go_binary")

go_binary(
  name = "hello",
  srcs = ["hello.go"],
)
```

* hello.go

```
```


## Hello binary with library

```bash
$ bazel build //cmd:hello
```

```
.
├── BUILD
├── WORKSPACE
├── cmd
│   ├── BUILD
│   └── hello.go
└── hello
    ├── BUILD
    └── hello.go
```

* BUILD

```py
load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library", "go_test")

go_binary(
  name = "hello",
  srcs = ["hello.go"],
  deps = [
  ],
)
```

* WORKSPACE

```py
workspace(name = "hello")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "842ec0e6b4fbfdd3de6150b61af92901eeb73681fd4d185746644c338f51d4c0",
    urls = [
        "https://storage.googleapis.com/bazel-mirror/github.com/bazelbuild/rules_go/releases/download/v0.20.1/rules_go-v0.20.1.tar.gz",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.20.1/rules_go-v0.20.1.tar.gz",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()
go_register_toolchains()
```
* cmd/BUILD

```py
load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library", "go_test")

go_binary(
  name = "hello",
  srcs = ["hello.go"],
  deps = [
    "//hello:hello",
  ],
)
```
* cmd/hello.go

```go
package main

import (
	"fmt"
	"github.com/si-you/bazel-golang-examples/hello"
)

func main() {
	fmt.Printf("%s\n", hello.Message())
}
```

* hello/BUILD

```py
package(default_visibility=["//visibility:public"])

load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library", "go_test")

go_library(
  name = "hello",
  srcs = [
    "hello.go",
  ],
  importpath = "github.com/si-you/bazel-golang-examples/hello",
)
```

* hello/hello.go

```go
package hello

func Message() string {
	return "Hello Bazel!"
}
```

## gazelle

gazelle generates BUILD.bazel files.

* origin directories

  ```
  .
  ├── BUILD.bazel
  ├── WORKSPACE
  └── src
      └── hello
          ├── cmd
          │   └── main.go
          └── hello.go
  ```

* BUILD.bazel

  ```py
  load("@bazel_gazelle//:def.bzl", "gazelle")

  gazelle(
      name = "gazelle",
      command = "fix",
  )
  ```

* WORKSPACE

  ```py

  ```

* src/hello/hello.go

  ```go
  package hello

  func Message() string {
    return "Hello Bazel!"
  }
  ```

* src/hello/cmd/main.go

  ```go
  package main

  import (
    "fmt"
    "hello"
  )

  func main() {
    fmt.Printf("%s\n", hello.Message())
  }
  ```

* generate bazel files

  ```bash
  $ bazel run -c opt //:gazelle
  ```

* after directoreis

  ```
  .
  ├── BUILD.bazel
  ├── README.md
  ├── WORKSPACE
  └── src
      ├── BUILD.bazel
      └── hello
          └── cmd
              ├── BUILD.bazel
              └── main.go
  ```

* src/BUILD.bazel

  ```py
  load("@io_bazel_rules_go//go:def.bzl", "go_library")

  go_library(
      name = "go_default_library",
      srcs = ["hello.go"],
      importpath = "",
      visibility = ["//visibility:public"],
  )
  ```

* src/hello/cmd/BUILD.bazel

  ```py
  load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

  go_library(
      name = "go_default_library",
      srcs = ["main.go"],
      importpath = "hello/cmd",
      visibility = ["//visibility:private"],
  )

  go_binary(
      name = "cmd",
      embed = [":go_default_library"],
      visibility = ["//visibility:public"],
  )
  ```

* build

  ```bash
  $ bazel build //hello/main:cmd
  ```

* "`no package error`"

```bash
$ bazel run //:gazelle -- update-repos github.com/grpc-ecosystem/go-grpc-prometheus
$ bazel run //:gazelle
$ bazel build //...
```

## go get

* origin structure

```
.
├── BUILD
├── WORKSPACE
└── redis.go
```

* build

```bash
$ bazel build //:redis --incompatible_disable_deprecated_attr_params=false
```

* WORKSPACE

```py
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "io_bazel_rules_go",
    urls = ["https://github.com/bazelbuild/rules_go/releases/download/0.13.0/rules_go-0.13.0.tar.gz"],
    sha256 = "ba79c532ac400cefd1859cbc8a9829346aa69e3b99482cd5a54432092cbc3933",
)
http_archive(
    name = "bazel_gazelle",
    urls = ["https://github.com/bazelbuild/bazel-gazelle/releases/download/0.13.0/bazel-gazelle-0.13.0.tar.gz"],
    sha256 = "bc653d3e058964a5a26dcad02b6c72d7d63e6bb88d94704990b908a1445b8758",
)
load("@io_bazel_rules_go//go:def.bzl", "go_rules_dependencies", "go_register_toolchains")
go_rules_dependencies()
go_register_toolchains()
load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies", "go_repository")
gazelle_dependencies()

# Repos for radix.
go_repository(
  name = "com_github_mediocregopher_radix_v2",
  # remote = "https://github.com/mediocregopher/radix.v2",
  importpath = "github.com/mediocregopher/radix.v2",
  commit = "596a3ed684d9390f4831d6800dffd6a6067933d5",
)
```

* BUILD

```py
load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library", "go_test")

go_binary(
  name = "redis",
  srcs = ["redis.go"],
  deps = [
    "@com_github_mediocregopher_radix_v2//redis:go_default_library",
  ],
)
```

* redis.go

```go
package main

import (
	"flag"
	"fmt"
	"github.com/mediocregopher/radix.v2/redis"
	"log"
)

var (
	redisAddr = flag.String("redis_addr", "localhost:6379", "Redis address.")
)

func main() {
	flag.Parse()

	c, err := redis.Dial("tcp", *redisAddr)
	if err != nil {
		log.Fatalf("Redis connection failed: %v", err)
	}

	resp := c.Cmd("SET", "hello", "world")
	if resp.Err != nil {
		log.Fatalf("Set message failed: %v", resp.Err)
	}
	fmt.Printf("SET completed: %v\n", resp)

	msg, err := c.Cmd("GET", "hello").Str()
	if err != nil {
		log.Fatalf("Get mesage failed: %v", err)
	}
	fmt.Printf("key: hello value: %v\n", msg)
}
```

## protobuf

* origin structure

```py 
.
├── WORKSPACE
├── cmd
│   ├── BUILD
│   └── hello.go
└── proto
    ├── BUILD
    └── greet.proto
```

* WORKSPACE

```py

```

* cmd/BUILD

```py
load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library", "go_test")

go_binary(
  name = "hello",
  srcs = ["hello.go"],
  deps = [
    "//proto:greet_go_proto",
  ],
)
```

* cmd/hello.go

```go
package main

import (
	"fmt"
	pb "github.com/si-you/examples/proto/greet"
)

func main() {
	p := pb.Greet{
		Greeting: "Hello",
		Name: "Bazel",
	}
	fmt.Printf("Greet proto: %v\n", p)
	fmt.Printf("%s, %s!\n", p.Greeting, p.Name)
}
```

* proto/BUILD

```py
package(default_visibility=["//visibility:public"])

load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")

proto_library(
    name = "greet_proto",
    srcs = ["greet.proto"],
)

go_proto_library(
    name = "greet_go_proto",
    compiler = "@io_bazel_rules_go//proto:go_grpc",
    importpath = "github.com/si-you/examples/proto/greet",
    proto = ":greet_proto",
)
```

* proto/greet.proto

```proto
syntax = "proto3";

package greet;

message Greet {
  string greeting = 1;
  string name = 2;
}
```

* build

```bash
$ bazel build //hello:cmd
```

* structure after build

```
.
├── README.md
├── WORKSPACE
├── cmd
│   ├── BUILD
│   └── hello.go
└── proto
    ├── BUILD
    └── greet.proto
```

## grpc

TODO

## go docker image

TODO