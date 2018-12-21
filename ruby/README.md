- [Abstract](#abstract)
- [Materials](#materials)
- [Basic Usages](#basic-usages)
  - [Hello World](#hello-world)
  - [REPL](#repl)
  - [Collections compared to c++ containers](#collections-compared-to-c-containers)
  - [Collections by examples](#collections-by-examples)
  - [Reserved Words](#reserved-words)
  - [Syntax](#syntax)
  - [Classes and Objects](#classes-and-objects)
  - [Variables](#variables)
  - [Operators](#operators)
  - [Decision makings](#decision-makings)
  - [Loops](#loops)
  - [Methods](#methods)
  - [Blocks](#blocks)
  - [Modules](#modules)
  - [Strings](#strings)
  - [Arrays](#arrays)
  - [Hashes](#hashes)
  - [Date & Time](#date--time)
  - [Ranges](#ranges)
  - [Iterators](#iterators)
  - [File I/O](#file-io)
  - [Exceptions](#exceptions)
- [Advanced](#advanced)
  - [Object Oriented](#object-oriented)
  - [Regular Expressions](#regular-expressions)
  - [Database Access](#database-access)
  - [Web Applications](#web-applications)
  - [Sending Email](#sending-email)
  - [Socket Programming](#socket-programming)
  - [Ruby/XML, XSLT](#rubyxml-xslt)
  - [Web Services](#web-services)
  - [Tk Guide](#tk-guide)
  - [Ruby/LDAP](#rubyldap)
  - [Multithreading](#multithreading)
  - [Built-in Functions](#built-in-functions)
  - [Predefined Variables](#predefined-variables)
  - [Predefined Constants](#predefined-constants)
  - [Associated Tools](#associated-tools)

------

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

## class, methods
class("")
"".methods
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
# define methods
# def method_name [( [arg [= default]]...[, * arg [, &expr ]])]
#    expr..
# end
# def method_name
#    expr..
# end
# def method_name (var1, var2)
#    expr..
# end
# def method_name (var1 = value1, var2 = value2)
#    expr..
# end
def test(a1 = "Ruby", a2 = "Perl")
   puts "The programming language is #{a1}"
   puts "The programming language is #{a2}"
end
test "C", "C++"
test

## return values
# return last statement
def test
   i = 100
   j = 10
   k = 0
end

def test
   i = 100
   j = 200
   k = 300
   return i, j, k
end
var = test
puts var
# 100
# 200
# 300

# variable number of parameters
def sample (*test)
   puts "The number of parameters is #{test.length}"
   for i in 0...test.length
      puts "The parameters are #{test[i]}"
   end
end
sample "Zara", "6", "F"
sample "Mac", "36", "M", "MCA"
# The number of parameters is 3
# The parameters are Zara
# The parameters are 6
# The parameters are F
# The number of parameters is 4
# The parameters are Mac
# The parameters are 36
# The parameters are M
# The parameters are MCA

## Class methods
class Accounts
   def reading_charge
   end
   def Accounts.return_date
   end
end
Accounts.return_date

## alias
alias foo bar
alias $MATCH $&

## undef
undef bar
```

## Blocks

block 은 yield 로 실행한다.

```ruby
## syntax
# block_name {
#    statement1
#    statement2
#    ..........
# }
def test
   puts "You are in the method"
   yield
   puts "You are again back to the method"
   yield
end
test {puts "You are in the block"}
# You are in the method
# You are in the block
# You are again back to the method
# You are in the block

# block argument
def test
   yield 5
   puts "You are in the method test"
   yield 100
end
test {|i| puts "You are in the block #{i}"}
# You are in the block 5
# You are in the method test
# You are in the block 100
```

## Modules

```ruby
## syntax
# module Identifier
#    statement1
#    statement2
#    ...........
# end

# Module defined in trig.rb file
module Trig
   PI = 3.141592654
   def Trig.sin(x)
   # ..
   end
   def Trig.cos(x)
   # ..
   end
end
# Module defined in moral.rb file
module Moral
   VERY_BAD = 0
   BAD = 1
   def Moral.sin(badness)
   # ...
   end
end

## require statement
# require 할때 현재 디렉토리를 기준으로 로드한다.
$LOAD_PATH << '.'

require 'trig.rb'
require 'moral'

y = Trig.sin(Trig::PI/4)
wrongdoing = Moral.sin(Moral::VERY_BAD)

## include statement
# support.rb
module Week
   FIRST_DAY = "Sunday"
   def Week.weeks_in_month
      puts "You have four weeks in a month"
   end
   def Week.weeks_in_year
      puts "You have 52 weeks in a year"
   end
end

# a.rb
$LOAD_PATH << '.'
require "support"

class Decade
include Week
   no_of_yrs = 10
   def no_of_months
      puts Week::FIRST_DAY
      number = 10*12
      puts number
   end
end
d1 = Decade.new
puts Week::FIRST_DAY
Week.weeks_in_month
Week.weeks_in_year
d1.no_of_months
# Sunday
# You have four weeks in a month
# You have 52 weeks in a year
# Sunday
# 120

## mixins
module A
   def a1
   end
   def a2
   end
end
module B
   def b1
   end
   def b2
   end
end

class Sample
include A
include B
   def s1
   end
end

samp = Sample.new
samp.a1
samp.a2
samp.b1
samp.b2
samp.s1
```

## Strings

```ruby
## expression substitution
x, y, z = 12, 36, 72
puts "The value of x is #{ x }."
puts "The sum of x and y is #{ x + y }."
puts "The average was #{ (x + y + z)/3 }."
# The value of x is 12.
# The sum of x and y is 48.
# The average was 40.

## general delimited strings
%{Ruby is fun.}  equivalent to "Ruby is fun."
%Q{ Ruby is fun. } equivalent to " Ruby is fun. "
%q[Ruby is fun.]  equivalent to a single-quoted string
%x!ls! equivalent to back tick command output `ls`

## character encoding
$KCODE = 'u'
# a : ASCII
# e : EUC
# n : same as ASCII
# u : UTF-8

## string built-in methods
myStr = String.new("THIS IS TEST")
foo = myStr.downcase
puts "#{foo}"
# this is test

## string unpack directives
"abc \0\0abc \0\0".unpack('A6Z6')   #=> ["abc", "abc "]
"abc \0\0".unpack('a3a3')           #=> ["abc", " \000\000"]
"abc \0abc \0".unpack('Z*Z*')       #=> ["abc ", "abc "]
"aa".unpack('b8B8')                 #=> ["10000110", "01100001"]
"aaa".unpack('h2H2c')               #=> ["16", "61", 97]
"\xfe\xff\xfe\xff".unpack('sS')     #=> [-2, 65534]
"now = 20is".unpack('M*')           #=> ["now is"]
"whole".unpack('xax2aX2aX1aX2a')    #=> ["h", "e", "l", "l", "o"]
```

## Arrays

```ruby
## creating arrays
names = Array.new(20)
puts names.size  # This returns 20
puts names.length # This also returns 20
#20
#20

names = Array.new(4, "mac")
puts "#{names}"
# ["mac", "mac", "mac", "mac"]

nums = Array.new(10) { |e| e = e * 2 }
puts "#{nums}"
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

nums = Array.[](1, 2, 3, 4, 5)
nums = Array[1, 2, 3, 4, 5]

digits = Array(0..9)
puts "#{digits}"
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

## array built-in methods
digits = Array(0..9)
num = digits.at(6)
puts "#{num}"
# 6

## array pack directives
a = [ "a", "b", "c" ]
n = [ 65, 66, 67 ]
puts a.pack("A3A3A3")   #=> "a  b  c  "
puts a.pack("a3a3a3")   #=> "a\000\000b\000\000c\000\000"
puts n.pack("ccc")      #=> "ABC"
# a  b  c
# abc
# ABC
```

## Hashes

```ruby
## creating hashes
months = Hash.new( "month" )
puts "#{months[0]}"
puts "#{months[72]}"
# month
# month
H = Hash["a" => 100, "b" => 200]
puts "#{H['a']}"
puts "#{H['b']}"
# 100
# 200

## built-in methods
$, = ", "
months = Hash.new( "month" )
months = {"1" => "January", "2" => "February"}
keys = months.keys
puts "#{keys}"
# ["1", "2"]
```

## Date & Time

```ruby
## Getting Current Date and Time
time1 = Time.new
puts "Current Time : " + time1.inspect
# Time.now is a synonym:
time2 = Time.now
puts "Current Time : " + time2.inspect
# Current Time : Mon Jun 02 12:02:39 -0700 2008
# Current Time : Mon Jun 02 12:02:39 -0700 2008

## getting components of a date & time
time = Time.new
# Components of a Time
puts "Current Time : " + time.inspect
puts time.year    # => Year of the date 
puts time.month   # => Month of the date (1 to 12)
puts time.day     # => Day of the date (1 to 31 )
puts time.wday    # => 0: Day of week: 0 is Sunday
puts time.yday    # => 365: Day of year
puts time.hour    # => 23: 24-hour clock
puts time.min     # => 59
puts time.sec     # => 59
puts time.usec    # => 999999: microseconds
puts time.zone    # => "UTC": timezone name
# Current Time : Mon Jun 02 12:03:08 -0700 2008
# 2008
# 6
# 2
# 1
# 154
# 12
# 3
# 8
# 247476
# UTC

## Time.utc, Time.gm and Time.local Functions
# July 8, 2008
Time.local(2008, 7, 8)  
# July 8, 2008, 09:10am, local time
Time.local(2008, 7, 8, 9, 10)   
# July 8, 2008, 09:10 UTC
Time.utc(2008, 7, 8, 9, 10)  
# July 8, 2008, 09:10:11 GMT (same as UTC)
Time.gm(2008, 7, 8, 9, 10, 11)  

time = Time.new
values = time.to_a
puts values
# [26, 10, 12, 2, 6, 2008, 1, 154, false, "MST"]

time = Time.new
values = time.to_a
puts Time.utc(*values)
# Mon Jun 02 12:15:36 UTC 2008

# Returns number of seconds since epoch
time = Time.now.to_i  
# Convert number of seconds into Time object.
Time.at(time)
# Returns second since epoch which includes microseconds
time = Time.now.to_f

## Timezones and daylight savings time
time = Time.new
# Here is the interpretation
time.zone       # => "UTC": return the timezone
time.utc_offset # => 0: UTC is 0 seconds offset from UTC
time.zone       # => "PST" (or whatever your timezone is)
time.isdst      # => false: If UTC does not have DST.
time.utc?       # => true: if t is in UTC time zone
time.localtime  # Convert to local timezone.
time.gmtime     # Convert back to UTC.
time.getlocal   # Return a new Time object in local zone
time.getutc     # Return a new Time object in UTC

## Formatting Times and Dates
time = Time.new
puts time.to_s
puts time.ctime
puts time.localtime
puts time.strftime("%Y-%m-%d %H:%M:%S")
Mon Jun 02 12:35:19 -0700 2008
Mon Jun  2 12:35:19 2008
Mon Jun 02 12:35:19 -0700 2008
2008-06-02 12:35:19

## Time Arithmetic
now = Time.now          # Current time
puts now
past = now - 10         # 10 seconds ago. Time - number => Time
puts past
future = now + 10  # 10 seconds from now Time + number => Time
puts future
diff = future - past     # => 10  Time - Time => number of seconds
puts diff
# Thu Aug 01 20:57:05 -0700 2013
# Thu Aug 01 20:56:55 -0700 2013
# Thu Aug 01 20:57:15 -0700 2013
# 20.0
```

## Ranges

```ruby
## rages as sequences
$, =", "   # Array value separator
range1 = (1..10).to_a
range2 = ('bar'..'bat').to_a
puts "#{range1}"
puts "#{range2}"
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# ["bar", "bas", "bat"]

# Assume a range
digits = 0..9
puts digits.include?(5)
ret = digits.min
puts "Min value is #{ret}"
ret = digits.max
puts "Max value is #{ret}"
ret = digits.reject {|i| i < 5 }
puts "Rejected values are #{ret}"
digits.each do |digit|
   puts "In Loop #{digit}"
end
# true
# Min value is 0
# Max value is 9
# Rejected values are 5, 6, 7, 8, 9
# In Loop 0
# In Loop 1
# In Loop 2
# In Loop 3
# In Loop 4
# In Loop 5
# In Loop 6
# In Loop 7
# In Loop 8
# In Loop 9

## ranges as conditions
score = 70
result = case score
   when 0..40 then "Fail"
   when 41..60 then "Pass"
   when 61..70 then "Pass with Merit"
   when 71..100 then "Pass with Distinction"
   else "Invalid Score"
end
puts result
# Pass with Merit

## Ranges as Intervals
if ((1..10) === 5)
   puts "5 lies in (1..10)"
end

if (('a'..'j') === 'c')
   puts "c lies in ('a'..'j')"
end

if (('a'..'j') === 'z')
   puts "z lies in ('a'..'j')"
end

# 5 lies in (1..10)
# c lies in ('a'..'j')
```

## Iterators

```ruby
## each iterator
# collection.each do |variable|
#    code
# end
ary = [1,2,3,4,5]
ary.each do |i|
   puts i
end
# 1
# 2
# 3
# 4
# 5

## collect iterator
# collection = collection.collect
a = [1,2,3,4,5]
b = Array.new
b = a.collect
puts b
# 1
# 2
# 3
# 4
# 5

a = [1,2,3,4,5]
b = a.collect{|x| 10*x}
puts b
# 10
# 20
# 30
# 40
# 50
```

## File I/O

```ruby
## puts statement
val1 = "This is variable one"
val2 = "This is variable two"
puts val1
puts val2
# This is variable one
# This is variable two

## gets statement
puts "Enter a value :"
val = gets
puts val
# Enter a value :
# This is entered value
# This is entered value

## putc statement
str = "Hello Ruby!"
putc str
# H

## print statement
print "Hello World"
print "Good Morning"
# Hello WorldGood Morning

## File.new
aFile = File.new("filename", "mode")
   # ... process the file
aFile.close

## File.open
File.open("filename", "mode") do |aFile|
   # ... process the file
end

## sysread
aFile = File.new("input.txt", "r")
if aFile
   content = aFile.sysread(20)
   puts content
else
   puts "Unable to open file!"
end

## syswrite
aFile = File.new("input.txt", "r+")
if aFile
   aFile.syswrite("ABCDEF")
else
   puts "Unable to open file!"
end

## each_byte
aFile = File.new("input.txt", "r+")
if aFile
   aFile.syswrite("ABCDEF")
   aFile.each_byte {|ch| putc ch; putc ?. }
else
   puts "Unable to open file!"
end
# s. .a. .s.i.m.p.l.e. .t.e.x.t. .f.i.l.e. .f.o.r. .t.e.s.t.i.n.g. .p.u.r.p.o.s.e...
# .
# .

## IO.readlines
arr = IO.readlines("input.txt")
puts arr[0]
puts arr[1]

## IO.foreach
IO.foreach("input.txt"){|block| puts block}

## renaming and deleting
# Rename a file from test1.txt to test2.txt
File.rename( "test1.txt", "test2.txt" )
# Delete file test2.txt
File.delete("test2.txt")

## file modes and ownership
file = File.new( "test.txt", "w" )
file.chmod( 0755 )

## file inquries
File.open("file.rb") if File::exists?( "file.rb" )
# This returns either true or false
File.file?( "text.txt" ) 
# a directory
File::directory?( "/usr/local/bin" ) # => true
# a file
File::directory?( "file.rb" ) # => false
File.readable?( "test.txt" )   # => true
File.writable?( "test.txt" )   # => true
File.executable?( "test.txt" ) # => false
File.zero?( "test.txt" )      # => true
File.size?( "text.txt" )     # => 1002
File::ftype( "test.txt" )     # => file
File::ctime( "test.txt" ) # => Fri May 09 10:06:37 -0700 2008
File::mtime( "text.txt" ) # => Fri May 09 10:44:44 -0700 2008
File::atime( "text.txt" ) # => Fri May 09 10:45:01 -0700 2008

## navigating through directories
Dir.chdir("/usr/bin")
puts Dir.pwd # This will return something like /usr/bin
puts Dir.entries("/usr/bin").join(' ')
Dir.foreach("/usr/bin") do |entry|
   puts entry
end
Dir["/usr/bin/*"]

## creaing a directory
Dir.mkdir("mynewdir")
Dir.mkdir( "mynewdir", 755 )

## deleting a directory
Dir.delete("testdir")

## creating files & temporary directories
require 'tmpdir'
   tempfilename = File.join(Dir.tmpdir, "tingtong")
   tempfile = File.new(tempfilename, "w")
   tempfile.puts "This is a temporary file"
   tempfile.close
   File.delete(tempfilename)
require 'tempfile'
   f = Tempfile.new('tingtong')
   f.puts "Hello"
   puts f.path
   f.close
```

## Exceptions

```ruby
## syntax
# begin  
# # -  
# rescue OneTypeOfException  
# # -  
# rescue AnotherTypeOfException  
# # -  
# else  
# # Other exceptions
# ensure
# # Always will be executed
# end
begin
   file = open("/unexistant_file")
   if file
      puts "File opened successfully"
   end
rescue
      file = STDIN
end
print file, "==", STDIN, "\n"

## using retry statement
# begin
#    # Exceptions raised by this code will 
#    # be caught by the following rescue clause
# rescue
#    # This block will capture all types of exceptions
#    retry  # This will move control to the beginning of begin
# end
begin
   file = open("/unexistant_file")
   if file
      puts "File opened successfully"
   end
rescue
   fname = "existant_file"
   retry
end

## using raise statement
# raise 
# OR
# raise "Error Message" 
# OR
# raise ExceptionType, "Error Message"
# OR
# raise ExceptionType, "Error Message" condition
begin  
   puts 'I am before the raise.'  
   raise 'An error has occurred.'  
   puts 'I am after the raise.'  
rescue  
   puts 'I am rescued.'  
end  
puts 'I am after the begin block.'  
# I am before the raise.  
# I am rescued.  
# I am after the begin block. 

begin  
   raise 'A test exception.'  
rescue Exception => e  
   puts e.message  
   puts e.backtrace.inspect  
end  
# A test exception.
# ["main.rb:4"]

## using ensure statement
# begin 
#    #.. process 
#    #..raise exception
# rescue 
#    #.. handle error 
# ensure 
#    #.. finally ensure execution
#    #.. This will always execute.
# end
begin
   raise 'A test exception.'
rescue Exception => e
   puts e.message
   puts e.backtrace.inspect
ensure
   puts "Ensuring execution"
end
# A test exception.
# ["main.rb:4"]
# Ensuring execution

## using else statement
# begin 
#    #.. process 
#    #..raise exception
# rescue 
#    # .. handle error
# else
#    #.. executes if there is no exception
# ensure 
#    #.. finally ensure execution
#    #.. This will always execute.
# end
begin
   # raise 'A test exception.'
   puts "I'm not raising exception"
rescue Exception => e
   puts e.message
   puts e.backtrace.inspect
else
   puts "Congratulations-- no errors!"
ensure
   puts "Ensuring execution"
end
# I'm not raising exception
# Congratulations-- no errors!
# Ensuring execution

## catch and throw
# throw :lablename
# #.. this will not be executed
# catch :lablename do
# #.. matching catch will be executed after a throw is encountered.
# end
# OR
# throw :lablename condition
# #.. this will not be executed
# catch :lablename do
# #.. matching catch will be executed after a throw is encountered.
# end
def promptAndGet(prompt)
   print prompt
   res = readline.chomp
   throw :quitRequested if res == "!"
   return res
end

catch :quitRequested do
   name = promptAndGet("Name: ")
   age = promptAndGet("Age: ")
   sex = promptAndGet("Sex: ")
   # ..
   # process information
end
promptAndGet("Name:")
# Name: Ruby on Rails
# Age: 3
# Sex: !
# Name:Just Ruby

## Class Exception
class FileSaveError < StandardError
   attr_reader :reason
   def initialize(reason)
      @reason = reason
   end
end

File.open(path, "w") do |file|
begin
   # Write out the data ...
rescue
   # Something went wrong!
   raise FileSaveError.new($!)
end
end
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

| variable | desc |
|:---------|:------|
| `$!` | The last exception object raised. The exception object can also be accessed using => in rescue clause. |
| `$@` |  |
| `$/` |  |
| `$\` |  |
| `$,` |  |
| `$;` |  |
| `$.` |  |
| `$<` |  |
| `$>` |  |
| `$0` | The name of the current Ruby program being executed. |
| `$$` |  |
| `$?` |  |
| `$:` |  |
| `$DEBUG` |  |
| `$defout` |  |
| `$F` |  |
| `$FILENAME` |  |
| `$LOAD_PATH` |  |
| `$SAFE` |  |
| `$stdin` |  |
| `$stderr` |  |
| `$VERBOSE` |  |
| `$-x` |  |
| `$-0` |  |
| `$-a` |  |
| `$-d` |  |
| `$-F` |  |
| `$-i` |  |
| `$-I` |  |
| `$-p` |  |
| `$_` |  |
| `$~` |  |
| `$n($1, $2, $3...)` |  |
| `$&` |  |
| ``` $` ``` |  |
| `$'` |  |
| `$+` |  |

## Predefined Constants

| constant | desc |
|:---------|:------|
| `TRUE` | true |
| `FALSE` |  |
| `NIL` |  |
| `ARGF` |  |
| `ARGV` |  |
| `DATA` |  |
| `ENV` |  |
| `RUBY_PLATFORM` |  |
| `RUBY_RELEASE_DATE` |  |
| `RUBY_VERSION` |  |
| `STDERR` |  |
| `STDIN` |  |
| `STDOUT` |  |
| `TOPLEVEL_BINDING` |  |

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

