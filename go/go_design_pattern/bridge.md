```go
package main

import "fmt"

// computer: Abstraction interface
type computer interface {
	print()
	setPrinter(printer)
}

// mac: Refined abstraction
type mac struct {
	printer printer
}

func (m *mac) print() {
	fmt.Println("Print request for mac")
	m.printer.printFile()
}

func (m *mac) setPrinter(p printer) {
	m.printer = p
}

// windows: Refined abstraction
type windows struct {
	printer printer
}

func (w *windows) print() {
	fmt.Println("Print request for windows")
	w.printer.printFile()
}

func (w *windows) setPrinter(p printer) {
	w.printer = p
}

// printer: Implementation interface
type printer interface {
	printFile()
}

// epson: Concrete implementation
type epson struct {
}

func (p *epson) printFile() {
	fmt.Println("Printing by a EPSON Printer")
}

// hp: Concrete implementation
type hp struct {
}

func (p *hp) printFile() {
	fmt.Println("Printing by a HP Printer")
}

// main
func main() {

	hpPrinter := &hp{}
	epsonPrinter := &epson{}

	macComputer := &mac{}

	macComputer.setPrinter(hpPrinter)
	macComputer.print()
	fmt.Println()

	macComputer.setPrinter(epsonPrinter)
	macComputer.print()
	fmt.Println()

	winComputer := &windows{}

	winComputer.setPrinter(hpPrinter)
	winComputer.print()
	fmt.Println()

	winComputer.setPrinter(epsonPrinter)
	winComputer.print()
	fmt.Println()
}

// Output:
//Print request for mac
//Printing by a HP Printer
//
//Print request for mac
//Printing by a EPSON Printer
//
//Print request for windows
//Printing by a HP Printer
//
//Print request for windows
//Printing by a EPSON Printer
```
