# Materials

* [Factory Method in Go](https://refactoring.guru/design-patterns/factory-method/go/example#example-0)

# Example

![](img/factorymethod.png)

```go
package main

import "fmt"

type API interface {
	Say(name string) string
}

func NewAPI(t int) API {
	if t == 1 {
		return &helloAPI{}
	} else if t == 2 {
		return &worldAPI{}
	}
	return nil
}

type helloAPI struct{}

func (h *helloAPI) Say(name string) string {
	return fmt.Sprintf("Hello, %s", name)
}

type worldAPI struct{}

func (w *worldAPI) Say(name string) string {
	return fmt.Sprintf("World, %s", name)
}

func main() {
	var api1 API = NewAPI(1)
	var api2 API = NewAPI(2)
	fmt.Println(api1.Say("David"))
	fmt.Println(api2.Say("David"))
}
```
