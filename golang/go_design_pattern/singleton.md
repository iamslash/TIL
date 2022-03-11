```go
package main

import (
	"fmt"
	"sync"
)

type singleton struct {
	name string
}

var instance *singleton
var once sync.Once

func GetInstance() *singleton {
	once.Do(func() {
		instance = &singleton{
			name: "david",
		}
	})
	return instance
}

func main() {
	fmt.Println(GetInstance() == GetInstance())
	v1 := 3
	v2 := 3
	p1 := &v1
	p2 := &v2
	fmt.Println(v1 == v2)
	fmt.Println(p1 == p2)
	fmt.Println(*p1 == *p2)
}
```
