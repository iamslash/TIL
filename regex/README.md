- [Materials](#materials)
- [Special pattern characters](#special-pattern-characters)
- [Quantifiers](#quantifiers)
- [Groups](#groups)
- [Assertions](#assertions)
- [Alternatives](#alternatives)
- [Character classes](#character-classes)
  - [Individual characters](#individual-characters)
  - [Ragnes](#ragnes)
  - [POSIX-like classes](#posix-like-classes)
  - [Escape characters](#escape-characters)

-----

# Materials

* [ECMAScript regular expressions pattern syntax](http://www.cplusplus.com/reference/regex/ECMAScript/)
* [regular expressions 101](https://regex101.com/)
  * Test PCRE, ECMAScript, Python, Golang regex.

# Special pattern characters

| characters          | description                           | example                                                    |
| ------------------- | ------------------------------------- | ---------------------------------------------------------- |
| `.`                 | not newline such as LF, CR, LS, PS    | `.+` matches '<u>a b c</u>'                                |
| `\t`                | horizontal tab (HT) same as `\u0009`  |                                                            |
| `\n`                | newline (LF) same as`\000A`           |                                                            |
| `\v`                | vertical tab (VT) same as `\u000B`    |                                                            |
| `\f`                | form feed (FF) same as `\uu000C`      | ``                                                         |
| `\r`                | carriage return (CR) same as `\u000D` | ``                                                         |
| `\c`<i>letter</i>   | control code                          | `cI` matches 'vertical<u>        </u>tab'                  |
| `\x`<i>hh</i>       | ASCII character                       | ``                                                         |
| `\u`<i>hhhh</i>     | unicode character                     | ``                                                         |
| `\0`                | null                                  | ``                                                         |
| `\int`              | backreference                         | ``                                                         |
| `\d`                | digit                                 | ``                                                         |
| `\D`                | not digit                             | ``                                                         |
| `\s`                | whitespace                            | ``                                                         |
| `\S`                | not whitespace                        | ``                                                         |
| `\w`                | word                                  | ``                                                         |
| `\W`                | not word                              | ``                                                         |
| `\`<i>character</i> | character                             | ``                                                         |
| `[class]`           | character class                       | `[abc]+` matches '<u>a</u> <u>bb</u> <u>ccc</u>'           |
| `[^class]`          | negated character class               | `[^abc]+` matches '<u>Anything </u>b<u>ut </u>abc<u>.</u>' |

# Quantifiers

| characters   | description         | example                                                                         |
| ------------ | ------------------- | ------------------------------------------------------------------------------- |
| `*`          | 0 or more           | `ba*` matches "a <u>ba</u> <u>baa</u> aaa <u>ba</u> <u>b</u>"                   |
| `+`          | 1 or more           | `a+` matches '<u>a</u> <u>aa</u> <u>aaa</u> <u>aaaa</u> b<u>a</u>b b<u>aa</u>b' |
| `?`          | 0 or 1              | `ba?` matches '<u>ba</u> <u>b</u> a'                                            |
| `{int}`      | int                 | `a{3}` matches 'a aa <u>aaa</u> <u>aaa</u>a'                                    |
| `{int,}`     | int or more         | `a{3,}` matches 'a aa <u>aaa</u> <u>aaaa</u> <u>aaaaaa</u>'                     |
| `{min, max}` | between min and max | `a{3,6}` matches 'a aa <u>aaa</u> <u>aaaa</u> <u>aaaaaa</u>aaaa'                |

# Groups

| characters       | description   | example                                                                                |
| ---------------- | ------------- | -------------------------------------------------------------------------------------- |
| `(subpattern)`   | group         | `(he)+` matches '<u>hehe</u>h <u>he</u> <u>he</u>h'                                    |
| `(?:subpattern)` | passive group | `(?:he)+` matches '<u>hehe</u>h <u>he</u> <u>he</u>h' but won't create a capture group |

# Assertions

| characters       | description         | example                                               |
| ---------------- | ------------------- | ----------------------------------------------------- |
| `^`              | beginning of line   | `^\w+` matches '<U>start</U> of string'               |
| `$`              | end of line         | `\w+$` matches 'start of <U>string</U>'               |
| `\b`             | Word boundary       | `d\b` matches 'wor<U>d</U> boundaries are od<u>d</u>' |
| `\B`             | Not a word boundary | `r\B` matches '<u>r</u>egex is <u>r</u>eally cool'    |
| `(?=subpattern)` | Positive lookahead  | `foo(?=bar)` matches '<u>foo</u>bar foobaz'           |
| `(?!subpattern)` | Negative lookahead  | `foo(?!bar)` matches 'foobar <u>foo</u>baz'           |

# Alternatives

* `(a|b)` matches '<u>b</u>e<u>a</u>ch'.

# Character classes

## Individual characters

* `[abc]+` matches '<u>a</u> <u>bb</u> <u>ccc</u>'.
* `[^abc]+` matches '<u>Anything </u>b<u>ut </u>abc<u>.</u>'.

## Ragnes

* `[a-z]+` matches "O<u>nly</u> <u>a</u>-<u>z</u>".
* `[a-zA-Z]+` matches "<u>abc</u>123<u>DEF</u>".
  
## POSIX-like classes

| class           | description           |
| --------------- | --------------------- |
| `[:classname:]` | character class       |
| `[.classname.]` | collating sequence    |
| `[=classname=]` | character equivalents |

| class        | description                        | equivalent (regex_traits) | example                                                                            |
| ------------ | ---------------------------------- | ------------------------- | ---------------------------------------------------------------------------------- |
| `[:alnum:]`  | alpha-numerical character          | `isalnum`                 |                                                                                    |
| `[:alpha:]`  | alphabetic character               | `isalpha`                 |                                                                                    |
| `[:blank:]`  | blank character                    | `isblank`                 |                                                                                    |
| `[:cntrl:]`  | control character                  | `iscntrl`                 |                                                                                    |
| `[:digit:]`  | decimal digit character            | `isdigit`                 | `[abc[:digit:]]` is a character class that matches a, b, c, or a digit             |
| `[:graph:]`  | character with graphical character | `isgraph`                 |                                                                                    |
| `[:lower:]`  | lowercase letter                   | `islower`                 |                                                                                    |
| `[:print:]`  | printable character                | `isprint`                 |                                                                                    |
| `[:punct:]`  | punctuation mark character         | `ispunct`                 |                                                                                    |
| `[:space:]`  | whitespace character               | `isspace`                 | `[^[:space:]]` is a character class that matches any character except a whitespace |
| `[:upper:]`  | uppercase character                | `isupper`                 |                                                                                    |
| `[:xdigit:]` | hexadecimal digit character        | `isxdigit`                |                                                                                    |
| `[:d:]`      | decimal digit character            | `isdigit`                 |                                                                                    |
| `[:w:]`      | word character                     | `isalnum`                 |                                                                                    |
| `[:s:]`      | whitespace character               | `isspace`                 |                                                                                    |

## Escape characters

???