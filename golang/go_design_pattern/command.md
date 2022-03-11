```go
package main

import "fmt"

// button: Invoker
type button struct {
	command command
}

func (b *button) press() {
	b.command.execute()
}

// command: Command interface
type command interface {
	execute()
}

// onCommand: Concrete command
type onCommand struct {
	device device
}

func (c *onCommand) execute() {
	c.device.on()
}

// offCommand: Concrete command
type offCommand struct {
	device device
}

func (c *offCommand) execute() {
	c.device.off()
}

// device: Receiver interface
type device interface {
	on()
	off()
}

// tv: Concrete receiver
type tv struct {
	isRunning bool
}

func (t *tv) on() {
	t.isRunning = true
	fmt.Println("Turning tv on")
}

func (t *tv) off() {
	t.isRunning = false
	fmt.Println("Turning tv off")
}

// main
func main() {
	tv := &tv{}

	onCommand := &onCommand{
		device: tv,
	}

	offCommand := &offCommand{
		device: tv,
	}

	onButton := &button{
		command: onCommand,
	}
	onButton.press()

	offButton := &button{
		command: offCommand,
	}
	offButton.press()
}
// Output:
//Turning tv on
//Turning tv off
```
