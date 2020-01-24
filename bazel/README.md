- [Abstract](#abstract)
- [Materials](#materials)
- [Tips](#tips)
- [Go Examples](#go-examples)
  - [Hello binary](#hello-binary)
  - [Hello binary with library](#hello-binary-with-library)
  - [Gazelle](#gazelle)
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

## Gazelle

* [gazelle](/gazelle/README.md)

----

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
├── BUILD
├── WORKSPACE
├── cmd
│   └── hello.go
└── greet
    └── greet.proto
```

* WORKSPACE

```py
workspace(name = "alpha")

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

http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "14ac30773fdb393ddec90e158c9ec7ebb3f8a4fd533ec2abbfd8789ad81a284b",
    strip_prefix = "rules_docker-0.12.1",
    urls = ["https://github.com/bazelbuild/rules_docker/releases/download/v0.12.1/rules_docker-v0.12.1.tar.gz"],
)

load("@io_bazel_rules_docker//repositories:repositories.bzl", container_repositories = "repositories")

container_repositories()

load("@io_bazel_rules_docker//repositories:deps.bzl", container_deps = "deps")

container_deps()

load("@io_bazel_rules_docker//container:container.bzl", "container_pull")

container_pull(
    name = "official_ubuntu",
    digest = "sha256:bc025862c3e8ec4a8754ea4756e33da6c41cba38330d7e324abd25c8e0b93300",
    registry = "index.docker.io",
    repository = "library/ubuntu",
    tag = "latest",
)

go_repository(
    name = "org_golang_google_grpc",
    build_file_proto_mode = "disable",
    importpath = "google.golang.org/grpc",
    sum = "h1:U4Oh9xJixJwAFqa1c5uLhNAh8ERM/lc3hNhPEJiAEhs=",
    version = "v1.22.3",
)

```

* cmd/BUILD

```py
load("@bazel_gazelle//:def.bzl", "gazelle")

# gazelle:prefix iamslash.com/alpha
gazelle(name = "gazelle")

```

* `cmd/hello.go`

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

* `gree/greet.proto`

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
$ go mod init iamslash.com/alpha
$ bazel run //:gazelle -- update-repos -from_file=go.mod
$ bazel run //:gazelle
$ bazel build //...
$ bazel test //...
$ bazel run //cmd:cmd
$ ./bazel-bin/cmd/darwin_amd64_stripped/cmd
```

* structure after build

```
.
├── BUILD
├── WORKSPACE
├── go.mod
├── go.sum
├── cmd
│   ├── BUILD.bazel
│   └── hello.go
└── proto
    ├── BUILD.bazel
    └── greet.proto
```

## gRPC

* structure 

```bash
.
├── BUILD
├── WORKSPACE
├── cmd
│   ├── client
│   │   └── client.go
│   └── server
│       └── server.go
└── proto
    └── helloworld
        └── helloworld.proto
```

* `BUILD`

```py
load("@bazel_gazelle//:def.bzl", "gazelle")

# gazelle:prefix iamslash.com/alpha
gazelle(name = "gazelle")

```

* `WORKSPACE`

```py
workspace(name = "alpha")

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

http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "14ac30773fdb393ddec90e158c9ec7ebb3f8a4fd533ec2abbfd8789ad81a284b",
    strip_prefix = "rules_docker-0.12.1",
    urls = ["https://github.com/bazelbuild/rules_docker/releases/download/v0.12.1/rules_docker-v0.12.1.tar.gz"],
)

load("@io_bazel_rules_docker//repositories:repositories.bzl", container_repositories = "repositories")

container_repositories()

load("@io_bazel_rules_docker//repositories:deps.bzl", container_deps = "deps")

container_deps()

load("@io_bazel_rules_docker//container:container.bzl", "container_pull")

container_pull(
    name = "official_ubuntu",
    digest = "sha256:bc025862c3e8ec4a8754ea4756e33da6c41cba38330d7e324abd25c8e0b93300",
    registry = "index.docker.io",
    repository = "library/ubuntu",
    tag = "latest",
)

```

* `cmd/client/client.go`

```go
package main

import (
	"log"
	"os"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/grpc"
	
	pb "iamslash.com/alpha/proto/helloworld"
)

const (
	address     = "localhost:50051"
	defaultName = "world"
)

func main() {
	// Set up a connection to the server.
	conn, err := grpc.Dial(address, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	c := pb.NewGreeterClient(conn)

	// Contact the server and print out its response.
	name := defaultName
	if len(os.Args) > 1 {
		name = os.Args[1]
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	r, err := c.SayHello(ctx, &pb.HelloRequest{Name: name})
	if err != nil {
		log.Fatalf("could not greet: %v", err)
	}
	log.Printf("Greeting: %s", r.Message)
}

```

* `cmd/server/server.go`

```go
package main

import (
	"log"
	"net"

	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"

	pb "iamslash.com/alpha/proto/helloworld"
)

const (
	port = ":50051"
)

// server is used to implement helloworld.GreeterServer.
type server struct{}

// SayHello implements helloworld.GreeterServer
func (s *server) SayHello(ctx context.Context, in *pb.HelloRequest) (*pb.HelloReply, error) {
	return &pb.HelloReply{Message: "Hello " + in.Name}, nil
}

func main() {
	lis, err := net.Listen("tcp", port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	s := grpc.NewServer()
	pb.RegisterGreeterServer(s, &server{})
	// Register reflection service on gRPC server.
	reflection.Register(s)
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}

```

* `proto/helloworld/helloworld.proto`

```go
syntax = "proto3";

package alpha.proto.helloworld;

// The greeting service definition.
service Greeter {
  // Sends a greeting
  rpc SayHello (HelloRequest) returns (HelloReply) {}
}

// The request message containing the user's name.
message HelloRequest {
  string name = 1;
}

// The response message containing the greetings
message HelloReply {
  string message = 1;
}

```

* build

```bash
$ go mod init iamslash.com/alpha
$ bazel run //:gazelle -- update-repos -from_file=go.mod
$ bazel run //:gazelle
# Build failed
$ bazel build //...
# Update dependencies of modules in WORKSPACE
$ bazel run //:gazelle -- update-repos -from_file=go.mod
$ bazel run //:gazelle
$ bazel build //...
$ bazel test //...
$ bazel run //cmd:server
$ bazel run //cmd:client
$ ./bazel-bin/cmd/darwin_amd64_stripped/server
$ ./bazel-bin/cmd/darwin_amd64_stripped/client
```

## go docker image

* structure

```bash
.
├── BUILD
├── WORKSPACE
└── cmd
    └── main
        └── hello.go
```

* `BUILD`

```py
load("@bazel_gazelle//:def.bzl", "gazelle")

# gazelle:prefix iamslash.com/alpha
gazelle(name = "gazelle")

```

* `WORKSPACE`

```py
workspace(name = "alpha")

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

http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "14ac30773fdb393ddec90e158c9ec7ebb3f8a4fd533ec2abbfd8789ad81a284b",
    strip_prefix = "rules_docker-0.12.1",
    urls = ["https://github.com/bazelbuild/rules_docker/releases/download/v0.12.1/rules_docker-v0.12.1.tar.gz"],
)

load("@io_bazel_rules_docker//repositories:repositories.bzl", container_repositories = "repositories")

container_repositories()

load("@io_bazel_rules_docker//repositories:deps.bzl", container_deps = "deps")

container_deps()

load("@io_bazel_rules_docker//container:container.bzl", "container_pull")

container_pull(
    name = "official_ubuntu",
    digest = "sha256:bc025862c3e8ec4a8754ea4756e33da6c41cba38330d7e324abd25c8e0b93300",
    registry = "index.docker.io",
    repository = "library/ubuntu",
    tag = "latest",
)

go_repository(
    name = "org_golang_google_grpc",
    build_file_proto_mode = "disable",
    importpath = "google.golang.org/grpc",
    sum = "h1:2dTRdpdFEEhJYQD8EMLB61nnrzSCTbG38PhqdhvOltg=",
    version = "v1.26.0",
)

go_repository(
    name = "com_github_golang_protobuf",
    importpath = "github.com/golang/protobuf",
    sum = "h1:6nsPYzhq5kReh6QImI3k5qWzO4PEbvbIW2cwSfR/6xs=",
    version = "v1.3.2",
)

```

* `cmd/main/hello.go`

```go
package main

import (
	"fmt"
)

func main() {
	fmt.Printf("Hello Bazel!\n")
}

```

* `cmd/main/BUILD.bazel`
  * add go_image for docker image

```py
load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")
load("@io_bazel_rules_docker//go:image.bzl", "go_image")

go_binary(
    name = "hello",
    embed = [":go_default_library"],
    visibility = ["//visibility:public"],
)

go_image(
    name = "hello_image",
    base = "@official_ubuntu//image",
    binary = ":hello",
)

go_library(
    name = "go_default_library",
    srcs = ["hello.go"],
    importpath = "iamslash.com/alpha/cmd/main",
    visibility = ["//visibility:private"],
)

```

* run

```bash
$ bazel run -c opt :hello_image
```

* build & load into docker

```bash
$ bazel build -c opt //cmd/main:hello_image.tar
$ sudo docker load -i bazel-bin/hello_image.tar
# Failed because of MacOS build
$ sudo docker run bazel/cmd/main:hello_image

```

* cross compile

```bash
$ bazel run -c opt \
  --platforms=@io_bazel_rules_go//go/toolchain:linux_amd64 \
  //cmd/main:hello_image
```

* build, load into docker, run docker image

```bash
$ bazel build \
  -c opt \
  --platforms=@io_bazel_rules_go//go/toolchain:linux_amd64 \
  cmd/main:hello_image.tar
$ docker load -i bazel-bin/cmd/main/hello_image.tar
$ docker run bazel/cmd/main:hello_image
```
