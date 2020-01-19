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

```bash
$ tree .

$ cat WORKSPACE

$ cat BUILD

# change WORKSPACE with deps of go.mod
$ bazel run //:gazelle -- update-repos -from_file=go.mod

# generate BUILD
$ bazel run //:gazelle

# build all
$ bazel build //...

# test all
$ bazel test //...
```
