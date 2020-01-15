# Materials

* [bazel-gazelle @ github](https://github.com/bazelbuild/bazel-gazelle)

# Basics

## Hello world with go build

updating...

## Hello World with gazel

```bash
$ tree .

$ cat WORKSPACE

$ cat BUILD

# change WORKSPACE with deps of go.mod
$ bazel run //:gazelle update-repos -from_file=go.mod

# generate BUILD
$ bazel run //:gazelle

# build all
$ bazel buil //...

# test all
$ bazel test //...
```
