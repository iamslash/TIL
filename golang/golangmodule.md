- [Materials](#materials)
- [Overview](#overview)
- [Basic Usages](#basic-usages)
  - [Creating a new module.](#creating-a-new-module)
  - [Adding a dependency](#adding-a-dependency)
  - [Upgrading dependencies](#upgrading-dependencies)
  - [Adding a dependency on a new major version](#adding-a-dependency-on-a-new-major-version)
  - [Upgrading a dependency to a new major version](#upgrading-a-dependency-to-a-new-major-version)
  - [Removing unused dependencies](#removing-unused-dependencies)
- [Advanced Usages](#advanced-usages)
  - [Cmd, Internal structure](#cmd-internal-structure)
  - [local modules](#local-modules)
- [How to publish go modules](#how-to-publish-go-modules)
- [How to update with v2](#how-to-update-with-v2)
- [How to migrate to Go modules](#how-to-migrate-to-go-modules)

----

# Materials

* [Using Go Modules](https://blog.golang.org/using-go-modules)
* [[Go] Go Modules 살펴보기](https://velog.io/@kimmachinegun/Go-Go-Modules-%EC%82%B4%ED%8E%B4%EB%B3%B4%EA%B8%B0-7cjn4soifk)
* [Everything you need to know about Packages in Go](https://medium.com/rungo/everything-you-need-to-know-about-packages-in-go-b8bac62b74cc)

# Overview

Handle module dependencies. Execute command line `go mod init` and this will make `go.mod`. Everytime you execute go command `go.mod` will be changed with new dependencies. 

`go.mod` have 4 kinds of directives `module, require, replace, exclude`

After `go build`, you can find out `go.sum` which has total dependent modules (name, version, sha1sum) and libries installed at `$GOPATH/pkg/mod/`.

# Basic Usages

## Creating a new module.

* directories

```bash
$ tree .
.
├── hello.go
└── hello_test.go
```

* `hello.go`

```go
package hello

func Hello() string {
	return "Hello, world."
}
```

* `hello_test.go`

```go
package hello

import "testing"

func TestHello(t *testing.T) {
	want := "Hello, world."
	if got := Hello(); got != want {
		t.Errorf("Hello() = %q, want %q", got, want)
	}
}
```

* run

```bash
$ go test
PASS
ok      _/D_/my/go/a    0.491s

# create go.mod file
$ go mod init iamslash.com/hello
go: creating new go.mod: module iamslash.com/hello

# go command will reflect dependencies in go.mod
$ go test
PASS
ok      iamslash.com/hello      0.494s

$ cat go.mod
module iamslash.com/hello

go 1.13
```

## Adding a dependency

* `hello.go`
  * added dependency with `rsc.io/quote`

```go
package hello

import "rsc.io/quote"

func Hello() string {
    return quote.Hello()
}
```

* run

```bash
# go command will reflect dependencies in go.mod
$ go test
go: downloading golang.org/x/text v0.0.0-20170915032832-14c0d48ead0c
go: extracting golang.org/x/text v0.0.0-20170915032832-14c0d48ead0c
go: finding rsc.io/sampler v1.3.0
go: finding golang.org/x/text v0.0.0-20170915032832-14c0d48ead0c
--- FAIL: TestHello (0.00s)
    hello_test.go:8: Hello() = "안녕, 세상.", want "Hello, world."
FAIL
exit status 1
FAIL    iamslash.com/hello      0.406s

# go.mod has been changed by go command
$ cat go.mod
module iamslash.com/hello

go 1.13

require rsc.io/quote v1.5.2

# Shows the current module and all it dependencies
$ go list -m all
iamslash.com/hello
golang.org/x/text v0.0.0-20170915032832-14c0d48ead0c
rsc.io/quote v1.5.2
rsc.io/sampler v1.3.0

# Shows cryptographic hashes of the content of specific module versions
$ cat go.sum
golang.org/x/text v0.0.0-20170915032832-14c0d48ead0c h1:qgOY6WgZOaTkIIMiVjBQcw93ERBE4m30iBm00nkL0i8=
golang.org/x/text v0.0.0-20170915032832-14c0d48ead0c/go.mod h1:NqM8EUOU14njkJ3fqMW+pc6Ldnwhi/IjpwHt7yyuwOQ=
rsc.io/quote v1.5.2 h1:w5fcysjrx7yqtD/aO+QwRjYZOKnaM9Uh2b40tElTs3Y=
rsc.io/quote v1.5.2/go.mod h1:LzX7hefJvL54yjefDEDHNONDjII0t9xZLPXsUe+TKr0=
rsc.io/sampler v1.3.0 h1:7uVkIFmeBqHfdjD+gZwtXXI+RODJ2Wc4O7MPEh/QiW4=
rsc.io/sampler v1.3.0/go.mod h1:T1hPZKmBbMNahiBKFy5HrXp6adAjACjK9JXDnKaTXpA=
```

## Upgrading dependencies

* go get

```bash
# Upgrade golang.org/x/text with latest version
$ go get golang.org/x/text
go: finding golang.org/x/text v0.3.0
go: downloading golang.org/x/text v0.3.0
go: extracting golang.org/x/text v0.3.0

# go command reflect dependencies in go.mod
$ go test
--- FAIL: TestHello (0.00s)
    hello_test.go:8: Hello() = "안녕, 세상.", want "Hello, world."
FAIL
exit status 1
FAIL    iamslash.com/hello      0.438s

$ go list -m all
iamslash.com/hello
golang.org/x/text v0.3.2
golang.org/x/tools v0.0.0-20180917221912-90fa682c2a6e
rsc.io/quote v1.5.2
rsc.io/sampler v1.3.0

# go.mod has been changed with upgraded golang.org/x/text
$ cat go.mod
module iamslash.com/hello

go 1.13

require (
	golang.org/x/text v0.3.2 // indirect
	rsc.io/quote v1.5.2
)

# Upgrade rsc.io/sampler
$ go get -v rsc.io/sampler
go: finding rsc.io/sampler v1.99.99
go: downloading rsc.io/sampler v1.99.99
go: extracting rsc.io/sampler v1.99.99
rsc.io/sampler

# Go command reflect depencies in go.mod
$ go test
--- FAIL: TestHello (0.00s)
    hello_test.go:8: Hello() = "99 bottles of beer on the wall, 99 bottles of beer, ...", want "Hello, world."
FAIL
exit status 1
FAIL    iamslash.com/hello      0.442s

# Shows versions of modules
$ go list -m -versions rsc.io/sampler
rsc.io/sampler v1.0.0 v1.2.0 v1.2.1 v1.3.0 v1.3.1 v1.99.99

# Downgrade a module
$ go get rsc.io/sampler@v1.3.1
go: finding rsc.io/sampler v1.3.1
go: downloading rsc.io/sampler v1.3.1
go: extracting rsc.io/sampler v1.3.1

$ go test
--- FAIL: TestHello (0.00s)
    hello_test.go:8: Hello() = "안녕, 세상.", want "Hello, world."
FAIL
exit status 1
FAIL    iamslash.com/hello      0.075s
```

## Adding a dependency on a new major version

* `hello.go`
  * added a new dependency with same module different major version.

```go
package hello

import (
    "rsc.io/quote"
    quoteV3 "rsc.io/quote/v3"
)

func Hello() string {
    return quote.Hello()
}

func Proverb() string {
    return quoteV3.Concurrency()
}
```

* `hello_test.go`

```go
func TestProverb(t *testing.T) {
    want := "Concurrency is not parallelism."
    if got := Proverb(); got != want {
        t.Errorf("Proverb() = %q, want %q", got, want)
    }
}
```

* run

```bash
$ go test
PASS
ok      iamslash.com/hello      0.507s

$ go list -m rsc.io/q...
rsc.io/quote v1.5.2
rsc.io/quote/v3 v3.1.0
```

* The convention such as `rsc.io/quote/v3` is [semantic import versioning](https://research.swtch.com/vgo-import). You can import same modules of different versions.

## Upgrading a dependency to a new major version

* We found a bug `rsc.io/quote`. We need to upgrade to `rsc.io/quote/v3`.

```bash
$ go doc rsc.io/quote/v3
package quote // import "rsc.io/quote/v3"

Package quote collects pithy sayings.

func Concurrency() string
func GlassV3() string
func GoV3() string
func HelloV3() string
func OptV3() string
```

* `hello.go`
  * Upgraded `rsc.io/quote` with `rsc.io/quote/v3`

```go
package hello

import quoteV3 "rsc.io/quote/v3"

func Hello() string {
    return quoteV3.HelloV3()
}

func Proverb() string {
    return quoteV3.Concurrency()
}
```

There is no need to use alias of module.

```go
package hello

import "rsc.io/quote/v3"

func Hello() string {
    return quote.HelloV3()
}

func Proverb() string {
    return quote.Concurrency()
}
```

* run

```bash
$ go test
```

## Removing unused dependencies

* run

```bash
# There is a unused module rsc.io/quote v1.5.2
$ go list -m all
iamslash.com/hello
golang.org/x/text v0.3.2
golang.org/x/tools v0.0.0-20180917221912-90fa682c2a6e
rsc.io/quote v1.5.2
rsc.io/quote/v3 v3.1.0
rsc.io/sampler v1.3.1

# Remove unused moduels
$ go mod tidy

# There is no rsc.io/quote v1.5.2
$ go list -m all
iamslash.com/hello
golang.org/x/text v0.3.2
golang.org/x/tools v0.0.0-20180917221912-90fa682c2a6e
rsc.io/quote/v3 v3.1.0
rsc.io/sampler v1.3.1

$ cat go.mod
module iamslash.com/hello

go 1.13

require (
	golang.org/x/text v0.3.2 // indirect
	rsc.io/quote/v3 v3.1.0
	rsc.io/sampler v1.3.1 // indirect
)

$ go test

```


# Advanced Usages

## Cmd, Internal structure

```bash
$ tree .
.
├── cmd
│   └── main
│       └── main.go
└── internal
    └── hello
        ├── hello.go
        └── hello_test.go
```

* `main.go`

```go
package main

import (
	"fmt"

	"iamslash.com/alpha/internal/hello"
)

func main() {
	fmt.Println(hello.Hello())
}
```

* `hello.go`

```go
package hello

func Hello() string {
	return "Hello, world."
}
```

* `hello_test.go`

```go
package hello

import "testing"

func TestHello(t *testing.T) {
	want := "Hello, world."
	if got := Hello(); got != want {
		t.Errorf("Hello() = %q, want %q", got, want)
	}
}
```

* run
  * There is no use of `replace` in go.mod to use local module.

```bash
$ go mod init iamslash.com/alpha

$ cat go.mod
module iamslash.com/alpha

go 1.13

# Show final versions that will be used in a build for all direct and indirect dependencies
$ go list -m all
iamslash.com/alpha

# Show available minor and patch upgrades for all direct and indirect dependencies
$ go list -u -m all
iamslash.com/alpha

# Update all direct and indirect dependencies to latest minor or patch upgrades
$ go get -u or go get -u=patch
go get or: malformed module path "or": missing dot in first path element
go get go: no Go source files
go get get: malformed module path "get": missing dot in first path element
go get -u=patch: malformed module path "-u=patch": leading dash

$ go build ./...

$ go test ./...
?   	iamslash.com/alpha/cmd/main	[no test files]
ok  	iamslash.com/alpha/internal/hello	0.006s

# Prune any no-longer-needed dependencies from go.mod and add any dependencies needed for other combinations of OS, architecture, and build tags
$ go mod tidy
```

## local modules

Use replace in `go.mod`.

```bash
$ cat go.mod
module iamslash.com/a

go 1.13

replace iamslash.com/a => ./
```

# How to publish go modules

Updating...

# How to update with v2

Updating...

# How to migrate to Go modules

Updating...
