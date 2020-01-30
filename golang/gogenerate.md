# Materials

* [Package generate](https://golang.org/pkg/cmd/go/internal/generate/)

# Basic Usages

* directory

```
.
├── cmd
│   └── main
│       └── main.go
├── go.mod
├── internal
│   └── foo
│       └── hello.go
└── main
```

* `cmd/main/main.go`

```go
package main

import (
	"fmt"

	"iamslash.com/alpha/internal/foo"
)

func main() {
	fmt.Println(foo.HelloWorld())
}
```

* `internal/foo/hello.go`

```go
package foo

// +build ignore
//go:generate echo "Hello World"

func HelloWorld() string {
	return "Hello World"
}
```

* build & generate

```bash
$ go mod init iamslash.com
$ go guild ./...
$ go generate
Hello World
```

* Q

  * Why `// +build ignore` doesn't work ??? 2020.01.31
  