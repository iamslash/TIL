- [Abstract](#abstract)
- [Materials](#materials)
- [Tips](#tips)
- [Go Examples](#go-examples)
  - [Hello binary](#hello-binary)
  - [Hello binary with library](#hello-binary-with-library)
  - [gazelle](#gazelle)
  - [Library Dependency](#library-dependency)
  - [Protobuf](#protobuf)
  - [gRPC](#grpc)
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

# Tips

* "`no package error`"

```bash
$ bazel run //:gazelle -- update-repos github.com/grpc-ecosystem/go-grpc-prometheus
$ bazel run //:gazelle
$ bazel build //...
```

# Go Examples

## Hello binary

```bash
$ bazel build //main:hello
```

```
.
├── BUILD
├── WORKSPACE
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

```go
package main

import "fmt"

func main() {
  fmt.Println("Hello World")
}
```

* build & run

```bash
# build
$ bazel build -c opt //:hello

# run
$ ./bazel-bin/darwin_amd64_stripped/hello
$ bazel run -c opt //:hello
```

## Hello binary with library

```bash
$ bazel build //cmd:hello
```

```
.
├── WORKSPACE
├── cmd
│   ├── BUILD
│   └── main.go
└── hello
    ├── BUILD
    └── hello.go
```

* `WORKSPACE`

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

* `cmd/BUILD`

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
* `cmd/hello.go`

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

* `hello/BUILD`

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

* `hello/hello.go`

```go
package hello

func Message() string {
	return "Hello Bazel!"
}
```

* build & run

```bash
# build
$ bazel build -c opt //...

# run
$ ./bazel-bin/cmd/darwin_amd64_stripped/hello
$ bazel run -c opt //cmd:hello
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

http_archive(
    name = "bazel_gazelle",
    sha256 = "41bff2a0b32b02f20c227d234aa25ef3783998e5453f7eade929704dcff7cd4b",
    urls = [
        "https://storage.googleapis.com/bazel-mirror/github.com/bazelbuild/bazel-gazelle/releases/download/v0.19.0/bazel-gazelle-v0.19.0.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.19.0/bazel-gazelle-v0.19.0.tar.gz",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains()

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")

gazelle_dependencies()
```

* BUILD.bazel

  ```py
  load("@bazel_gazelle//:def.bzl", "gazelle")

  gazelle(
      name = "gazelle",
      command = "fix",
  )
  ```

* `src/hello/hello.go`

  ```go
  package hello

  func Message() string {
    return "Hello Bazel!"
  }
  ```

* `src/hello/cmd/main.go`

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
  ├── WORKSPACE
  └── src
      ├── BUILD.bazel
      └── hello
          └── cmd
              ├── BUILD.bazel
              └── main.go
  ```

* `src/BUILD.bazel`

```py
# gazelle:prefix
```

* `src/hello/BUILD.bazel`

```py
load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["hello.go"],
    importpath = "hello",
    visibility = ["//visibility:public"],
)
```

* `src/hello/cmd/BUILD.bazel`

```py
load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["main.go"],
    importpath = "hello/cmd",
    visibility = ["//visibility:private"],
    deps = ["//src/hello:go_default_library"],
)

go_binary(
    name = "cmd",
    embed = [":go_default_library"],
    visibility = ["//visibility:public"],
)
```

* build

```bash
$ bazel build //src/hello/cmd:cmd
$ bazel build //...
$ bazel run //src/hello/cmd:cmd
$ ./bazel-bin/src/hello/cmd/darwin_amd64_stripped/cmd
```

## Library Dependency

* origin structure

```
.
├── BUILD
├── WORKSPACE
└── redis.go
```

* `BUILD`

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

* `WORKSPACE`

```py
workspace(name = "hello")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

http_archive(
    name = "com_google_protobuf",
    sha256 = "7c99ddfe0227cbf6a75d1e75b194e0db2f672d2d2ea88fb06bdc83fe0af4c06d",
    strip_prefix = "protobuf-3.9.2",
    urls = ["https://github.com/protocolbuffers/protobuf/releases/download/v3.9.2/protobuf-all-3.9.2.tar.gz"],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "842ec0e6b4fbfdd3de6150b61af92901eeb73681fd4d185746644c338f51d4c0",
    urls = [
        "https://storage.googleapis.com/bazel-mirror/github.com/bazelbuild/rules_go/releases/download/v0.20.1/rules_go-v0.20.1.tar.gz",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.20.1/rules_go-v0.20.1.tar.gz",
    ],
)

http_archive(
    name = "bazel_gazelle",
    sha256 = "41bff2a0b32b02f20c227d234aa25ef3783998e5453f7eade929704dcff7cd4b",
    urls = [
        "https://storage.googleapis.com/bazel-mirror/github.com/bazelbuild/bazel-gazelle/releases/download/v0.19.0/bazel-gazelle-v0.19.0.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.19.0/bazel-gazelle-v0.19.0.tar.gz",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

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

* `redis.go`

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

* build & run

```bash
$ bazel build //...
```

## Protobuf

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

## gRPC

TODO

## go docker image

TODO