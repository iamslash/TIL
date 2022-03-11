```go
package main

import "fmt"

// pizza: Component interface
type pizza interface {
	getPrice() int
}

// veggieMania: Concrete component
type veggieMania struct {
}

func (p *veggieMania) getPrice() int {
	return 15
}

// tomatorTopping: Concrete decorator
type tomatoTopping struct {
	pizza pizza
}

func (c *tomatoTopping) getPrice() int {
	pizzaPrice := c.pizza.getPrice()
	return pizzaPrice + 7
}

// cheeseTopping: Concrete decorator
type cheeseTopping struct {
	pizza pizza
}

func (c *cheeseTopping) getPrice() int {
	pizzaPrice := c.pizza.getPrice()
	return pizzaPrice + 10
}

// main
func main() {

	pizza := &veggieMania{}

	//Add cheese topping
	pizzaWithCheese := &cheeseTopping{
		pizza: pizza,
	}

	//Add tomato topping
	pizzaWithCheeseAndTomato := &tomatoTopping{
		pizza: pizzaWithCheese,
	}

	fmt.Printf("Price of veggeMania with tomato and cheese topping is %d\n", pizzaWithCheeseAndTomato.getPrice())
}
```
