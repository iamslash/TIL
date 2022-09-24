```go
package main

import "fmt"

// collection: Collection
type collection interface {
	createIterator() iterator
}

// userCollection: Concrete collection
type userCollection struct {
	users []*user
}

func (u *userCollection) createIterator() iterator {
	return &userIterator{
		users: u.users,
	}
}

// iterator: Iterator
type iterator interface {
	hasNext() bool
	getNext() *user
}

// userIterator: Concrete iterator
type userIterator struct {
	index int
	users []*user
}

func (u *userIterator) hasNext() bool {
	if u.index < len(u.users) {
		return true
	}
	return false

}
func (u *userIterator) getNext() *user {
	if u.hasNext() {
		user := u.users[u.index]
		u.index++
		return user
	}
	return nil
}

// user
type user struct {
	name string
	age  int
}

// main
func main() {

	user1 := &user{
		name: "a",
		age:  30,
	}
	user2 := &user{
		name: "b",
		age:  20,
	}

	userCollection := &userCollection{
		users: []*user{user1, user2},
	}

	iterator := userCollection.createIterator()

	for iterator.hasNext() {
		user := iterator.getNext()
		fmt.Printf("User is %+v\n", user)
	}
}
// Output:
//User is &{name:a age:30}
//User is &{name:b age:20}
```
