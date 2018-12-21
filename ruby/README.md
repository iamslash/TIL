# Abstract 

루비는 smalltalk 와 같은 pure object oriented 언어이다.

# Materials

* [learn ruby in y minutes](https://learnxinyminutes.com/docs/ruby/)
  * 짧은 시간에 ruby 가 어떠한 언어인지 파악해보자
* [루비 20분 가이드](https://www.ruby-lang.org/ko/documentation/quickstart/)
* [ruby @ tutorialspoint](https://www.tutorialspoint.com/ruby/)

# Basic Usages

## Hello World

* a.rb

```ruby
puts "hello world"
# ruby a.rb
```

## REPL

```bash
> irb
```

## Collections compared to c++ containers

| c++                  | ruby                   | 
|:---------------------|:-----------------------|
| `if, else`           | `if, else, elsif, end` |
| `for, while`         | `while, until, for`    |
| `array`              | ``              |
| `vector`             | ``              |
| `deque`              | ``                   |
| `forward_list`       | ``                   |
| `list`               | ``               |
| `stack`              | ``                   |
| `queue`              | ``                   |
| `priority_queue`     | ``               |
| `set`                | ``                   |
| `multiset`           | ``                   |
| `map`                | ``                   |
| `multimap`           | ``                   |
| `unordered_set`      | ``                   |
| `unordered_multiset` | ``                   |
| `unordered_map`      | ``                |
| `unordered_multimap` | ``                   |

## Collections by examples

* Array
* Queue
* Hash

## Reserved Words

```ruby
BEGIN do next then
END else nil true
alias elsif not undef
and end or unless
begin ensure redo until
break false rescue when
case for retry while
class if return while
def in self __FILE__
defined? module super __LINE__
```
## Syntax

```ruby
############################################################
# Here Document 
print <<EOF
  This is the first way of creating
  here document ie. multiple line string.
EOF

# same as above
print <<"EOF";
  This is the first way of creating
  here document ie. multiple line string.
EOF

# execute commands
print <<'EOC'
  echo hi there
  echo lo there
EOC

# you can stack them
print <<"foo", <<"bar"
    I said foo.
foo
    I said bar.
bar

#    This is the first way of creating
#    her document ie. multiple line string.
#    This is the second way of creating
#    her document ie. multiple line string.
# hi there
# lo there
#       I said foo.
#       I said bar.

#######################################################
# BEGIN statement
#   Declares code to be called before the program is run.
#
# BEGIN {
#     code
# }
puts "This is main Ruby program"
BEGIN {
    puts "initializing Ruby Program"
}
# Initializing Ruby Program
# This is main Ruby Program

# END statement
#   Declares code to be called at the end of the program
puts "This is main Ruby Program"
END {
    puts "Terminating Ruby Program"
}
BEGIN {
    puts "Initializing Ruby Program"
}
# Initializing Ruby Program
# This is main Ruby Program
# Terminating Ruby Program

=begin
This is a comment.
=end
```

## Classes and Objects

* local variables : `lowercase letters or _` 
* instance variables : `@`
* class variables : `@@`
* global variables : `$`

```ruby
## define class
class Customer
  @@no_of_customers = 0 # class variable
end
c1 = Customer.new
c2 = Customer.new

## initialize method
class Customer
  @@no_of_customers = 0
  def initialize(id, name, addr)
    @cust_id   = id    # instance variable
    @cust_name = name
    @cust_addr = addr
  end
end
c1 = Customer.new("1", "John", "Wisdom Apartments, Ludhiya")
c2 = Customer.new("2", "Poul", "New Empire road, Khandala")

## member functions
#    usually starts with a lowercase letter
# class Sample
#   def function
#     statement 1
#     statement 2
#   end
# end
class Sample
  def hello
    puts "Hello Ruby!"
  end
end
o = Sample.new
o.hello
```

## Variables

```ruby
## global variables
$global_variable = 10
class Class1
  def print_global
    puts("Global variable in Class1 is #$global_variable")
  end
end
class Class2
  def print_global
    puts("Global variable in Class2 is #$global_variable")
  end
end
c1 = Class1.new
c1.print_global
c2 = Class2.new
c2.print_global
# Global variable in Class1 is 10
# Global variable in Class2 is 10

## instance variables
class Customer
   def initialize(id, name, addr)
      @cust_id = id
      @cust_name = name
      @cust_addr = addr
   end
   def display_details()
      puts "Customer id #@cust_id"
      puts "Customer name #@cust_name"
      puts "Customer address #@cust_addr"
   end
end

# Create Objects
cust1 = Customer.new("1", "John", "Wisdom Apartments, Ludhiya")
cust2 = Customer.new("2", "Poul", "New Empire road, Khandala")

# Call Methods
cust1.display_details()
cust2.display_details()
# Customer id 1
# Customer name John
# Customer address Wisdom Apartments, Ludhiya
# Customer id 2
# Customer name Poul
# Customer address New Empire road, Khandala

## class variables
class Customer
   @@no_of_customers = 0
   def initialize(id, name, addr)
      @cust_id = id
      @cust_name = name
      @cust_addr = addr
   end
   def display_details()
      puts "Customer id #@cust_id"
      puts "Customer name #@cust_name"
      puts "Customer address #@cust_addr"
   end
   def total_no_of_customers()
      @@no_of_customers += 1
      puts "Total number of customers: #@@no_of_customers"
   end
end

# Create Objects
cust1 = Customer.new("1", "John", "Wisdom Apartments, Ludhiya")
cust2 = Customer.new("2", "Poul", "New Empire road, Khandala")

# Call Methods
cust1.total_no_of_customers()
cust2.total_no_of_customers()
# Total number of customers: 1
# Total number of customers: 2

## Constants
#    usually starts with a uppercase letter
class Example
   VAR1 = 100
   VAR2 = 200
   def show
      puts "Value of first Constant is #{VAR1}"
      puts "Value of second Constant is #{VAR2}"
   end
end

# Create Objects
object = Example.new()
object.show
# Value of first Constant is 100
# Value of second Constant is 200

## pseudo variables
# self, true, false, nil, __FILE__, __LINE__

## Integer Numbers
123                  # Fixnum decimal
1_234                # Fixnum decimal with underline
-500                 # Negative Fixnum
0377                 # octal
0xff                 # hexadecimal
0b1011               # binary
?a                   # character code for 'a'
?\n                  # code for a newline (0x0a)
12345678901234567890 # Bignum

## Floating Numbers
123.4                # floating point value
1.0e6                # scientific notation
4E20                 # dot not required
4e+20                # sign before exponential

## String Literals
puts 'escape using "\\"';
puts 'That\'s right';
# escape using "\"
# That's right
puts "Multiplication Value : #{24*60*60}";
# Multiplication Value : 86400

## Arrays
ary = [  "fred", 10, 3.14, "This is a string", "last element", ]
ary.each do |i|
   puts i
end
# fred
# 10
# 3.14
# This is a string
# last element

## Hashes
hsh = colors = { "red" => 0xf00, "green" => 0x0f0, "blue" => 0x00f }
hsh.each do |key, value|
   print key, " is ", value, "\n"
end
# red is 3840
# green is 240
# blue is 15

## Ranges
(10..15).each do |n| 
   print n, ' ' 
end
# 10 11 12 13 14 15
```

## Operators

```ruby
## Arithmetic Operators
# +
# -
# *
# /
# %
# **

## Comparison Operators
# ==
# !=
# >
# <
# >=
# <=
# # -1 if first operand is greater than 2nd operand
# # 1 if first operand is lesser than 2nd operand
# # 0 if same
# <=>
# # used to test equality within a when clause of a case statement 
# ===
# # same type and same value???
# .eql?
# # have the same object id???
# equal?

## Assignment Operators
# =
# +=
# -=
# *=
# /=
# %=
# **=

## bigtwise operators
# &
# |
# ^
# ~
# <<
# >>

## Logical Operators
# and
# or
# &&
# ||
# !
# not

## Ternary Operator
# ? :

## Range Operators
#..
1..10 # 1 to 10
#...
1...10 # 1 to 9

## defined? Operators
foo = 42
defined? foo    # => "local-variable"
defined? $_     # => "global-variable"
defined? bar    # => nil (undefined)

defined? puts        # => "method"
defined? puts(bar)   # => nil (bar is not defined here)
defined? unpack      # => nil (not defined here)

defined? super     # => "super" (if it can be called)
defined? super     # => nil (if it cannot be)

defined? yield    # => "yield" (if there is a block passed)
defined? yield    # => nil (if there is no block)

## . :: Operators
MR_COUNT = 0         # constant defined on main Object class
module Foo
   MR_COUNT = 0
   ::MR_COUNT = 1    # set global count to 1
   MR_COUNT = 2      # set local count to 2
end
puts MR_COUNT        # this is the global constant
puts Foo::MR_COUNT   # this is the local "Foo" constant

CONST = ' out there'
class Inside_one
   CONST = proc {' in there'}
   def where_is_my_CONST
      ::CONST + ' inside one'
   end
end
class Inside_two
   CONST = ' inside two'
   def where_is_my_CONST
      CONST
   end
end
puts Inside_one.new.where_is_my_CONST
puts Inside_two.new.where_is_my_CONST
puts Object::CONST + Inside_two::CONST
puts Inside_two::CONST + CONST
puts Inside_one::CONST
puts Inside_one::CONST.call + Inside_two::CONST
```

## Decision makings

```ruby
## if...else
#
# if conditional [then]
#    code...
# [elsif conditional [then]
#    code...]...
# [else
#    code...]
# end
x = 1
if x > 2
   puts "x is greater than 2"
elsif x <= 2 and x!=0
   puts "x is 1"
else
   puts "I can't guess the number"
end
# x is 1

## if modifier
#
# code if condition
$debug = 1
print "debug\n" if $debug
# debug

## unless statement
#
# unless conditional [then]
#    code
# [else
#    code ]
# end
x = 1 
unless x>=2
   puts "x is less than 2"
 else
   puts "x is greater than 2"
end
# x is less than 2

## unless modifier
#
# code unless conditional
$var =  1
print "1 -- Value is set\n" if $var
print "2 -- Value is set\n" unless $var
$var = false
print "3 -- Value is set\n" unless $var
# 1 -- Value is set
# 3 -- Value is set

## case statement
#
# case expression
# [when expression [, expression ...] [then]
#    code ]...
# [else
#    code ]
# end
#
# case expr0
# when expr1, expr2
#    stmt1
# when expr3, expr4
#    stmt2
# else
#    stmt3
# end
#
# _tmp = expr0
# if expr1 === _tmp || expr2 === _tmp
#    stmt1
# elsif expr3 === _tmp || expr4 === _tmp
#    stmt2
# else
#    stmt3
# end
$age =  5
case $age
when 0 .. 2
   puts "baby"
when 3 .. 6
   puts "little child"
when 7 .. 12
   puts "child"
when 13 .. 18
   puts "youth"
else
   puts "adult"
end
```

## Loops

```ruby
## while statement
# while conditional [do]
#    code
# end
$i = 0
$num = 5

while $i < $num  do
   puts("Inside the loop i = #$i" )
   $i +=1
end
# Inside the loop i = 0
# Inside the loop i = 1
# Inside the loop i = 2
# Inside the loop i = 3
# Inside the loop i = 4

## while modifier
#
# code while condition
# OR
# begin 
#   code 
# end while conditional
$i = 0
$num = 5
begin
   puts("Inside the loop i = #$i" )
   $i +=1
end while $i < $num
# Inside the loop i = 0
# Inside the loop i = 1
# Inside the loop i = 2
# Inside the loop i = 3
# Inside the loop i = 4

## until statement
#
# until conditional [do]
#    code
# end
$i = 0
$num = 5

until $i > $num  do
   puts("Inside the loop i = #$i" )
   $i +=1;
end
Inside the loop i = 0
Inside the loop i = 1
Inside the loop i = 2
Inside the loop i = 3
Inside the loop i = 4
Inside the loop i = 5

## until modifier
# code until conditional
# OR
# begin
#    code
# end until conditional
$i = 0
$num = 5
begin
   puts("Inside the loop i = #$i" )
   $i +=1;
end until $i > $num
# Inside the loop i = 0
# Inside the loop i = 1
# Inside the loop i = 2
# Inside the loop i = 3
# Inside the loop i = 4
# Inside the loop i = 5

## for statement
# for variable [, variable ...] in expression [do]
#    code
# end
for i in 0..5
   puts "Value of local variable is #{i}"
end
# Value of local variable is 0
# Value of local variable is 1
# Value of local variable is 2
# Value of local variable is 3
# Value of local variable is 4
# Value of local variable is 5

# (expression).each do |variable[, variable...]| code end
(0..5).each do |i|
   puts "Value of local variable is #{i}"
end

## break statement
for i in 0..5
   if i > 2 then
      break
   end
   puts "Value of local variable is #{i}"
end
# Value of local variable is 0
# Value of local variable is 1
# Value of local variable is 2

## next statement
for i in 0..5
   if i < 2 then
      next
   end
   puts "Value of local variable is #{i}"
end
# Value of local variable is 2
# Value of local variable is 3
# Value of local variable is 4
# Value of local variable is 5

## redo statement
for i in 0..5
   if i < 2 then
      puts "Value of local variable is #{i}"
      redo
   end
end
# Value of local variable is 0
# Value of local variable is 0
# ............................

## retry statement
# begin
#    do_something # exception raised
# rescue
#    # handles error
#    retry  # restart from beginning
# end
# for i in 1..5
#    retry if some_condition # restart from i == 1
# end
for i in 0..5
  retry if i > 2
  puts "Value of local variable is #{i}"
end
# Value of local variable is 1
# Value of local variable is 2
# Value of local variable is 1
# Value of local variable is 2
# Value of local variable is 1
# Value of local variable is 2
# ............................
```

## Methods

```ruby
```

## Blocks

```ruby
```

## Modules

```ruby
```

## Strings

```ruby
```

## Arrays

```ruby
```

## Hashes

```ruby
```

## Date & Time

```ruby
```

## Ranges

```ruby
```

## Iterators

```ruby
```

## File I/O

```ruby
```

## Exceptions

```ruby
```

# Advanced

## Object Oriented

```ruby
```

## Regular Expressions

```ruby
```

## Database Access

```ruby
```

## Web Applications

```ruby
```

## Sending Email

```ruby
```

## Socket Programming

```ruby
```

## Ruby/XML, XSLT

```ruby
```

## Web Services

```ruby
```

## Tk Guide

```ruby
```

## Ruby/LDAP

```ruby
```

## Multithreading

```ruby
```

## Built-in Functions

```ruby
```

## Predefined Variables

```ruby
```

## Predefined Constants

```ruby
```

## Associated Tools

* Ruby Gems
  * package manager
* Ruby Debugger
  * debugger like gdb
* Interative Ruby (irb)
  * REPL like ipython
* Ruby Profiler
* eRuby
  * embedded Ruby ???
* ri
  * ruby interactive reference  

