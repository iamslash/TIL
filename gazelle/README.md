- [Materials](#materials)
- [Basics](#basics)
  - [Hello world with go build](#hello-world-with-go-build)
  - [Hello World with bazel](#hello-world-with-bazel)
  - [Hello World with gazel](#hello-world-with-gazel)

----

# Materials

* [bazel-gazelle @ github](https://github.com/bazelbuild/bazel-gazelle)

# Basics

## Hello world with go build

* [golang build](/golang/README.md#go-build)

## Hello World with bazel

* [bazel](/bazel/README.md)

## Hello World with gazel

* structure

```bash
$ tree .
.
├── BUILD
├── WORKSPACE
├── cmd
│   └── main
│       ├── BUILD.bazel
│       └── main.go
└── go.mod
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

* `BUILD`

```py
load("@bazel_gazelle//:def.bzl", "gazelle")

# gazelle:prefix iamslash.com/alpha
gazelle(name = "gazelle")
```

* `cmd/main/main.go`

```go
package main

import "fmt"

func main() {
	fmt.Println("Hello World")
}
```

* go mod init

```bash
# This will generate go.mod
$ go mod init iamslash.com/alpha
```

* run gazelle

```bash
# This will generate cmd/main/BUILD.bazel
$ bazel run //:gazelle
```

* build with bazel

```bash
$ bazel build //...

$ bazel test //...

$ bazel run //cmd/main:main
```

* update dependencies

```bash
# change WORKSPACE with a dep
$ bazel run //:gazelle -- update-repos gopkg.in/yaml.v2

# change WORKSPACE with deps of go.mod
$ bazel run //:gazelle -- update-repos -from_file=go.mod
```
