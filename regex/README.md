- [Materials](#materials)
- [Special Pattern Characters](#special-pattern-characters)
- [Quantifiers](#quantifiers)
- [Groups](#groups)
- [Assertions](#assertions)
- [Alternatives](#alternatives)
- [Character Classes](#character-classes)
  - [Individual Characters](#individual-characters)
  - [Ranges](#ranges)
  - [POSIX-like Classes](#posix-like-classes)
  - [Escape Characters](#escape-characters)
- [Useful Regex Examples](#useful-regex-examples)
  - [URL Parsing](#url-parsing)
  - [Phone Number Formatting](#phone-number-formatting)
  - [Add Commas to Numbers](#add-commas-to-numbers)
  - [Validate Email](#validate-email)
  - [Validate Password](#validate-password)
  - [Remove HTML Tags](#remove-html-tags)
  - [Validate IP Addresses](#validate-ip-addresses)

---

## Materials
* [ECMAScript Regular Expressions Pattern Syntax](http://www.cplusplus.com/reference/regex/ECMAScript/)
* [Regex101](https://regex101.com/): Test PCRE, ECMAScript, Python, Golang regex.

---

## Special Pattern Characters

| Character         | Description                                   | Example                                                                 |
| ----------------- | --------------------------------------------- | ----------------------------------------------------------------------- |
| `.`               | Matches any character except newline         | `a.c` matches `abc`, `aXc`, but not `ac`.                              |
| `\t`             | Horizontal tab (HT)                          | Matches a tab character.                                               |
| `\n`             | Newline (LF)                                 | Matches a line feed character.                                         |
| `\r`             | Carriage return (CR)                         | Matches a carriage return character.                                   |
| `\d`             | Digit (0-9)                                  | Equivalent to `[0-9]`. Matches `1`, `5`, etc.                          |
| `\D`             | Not a digit                                  | Equivalent to `[^0-9]`. Matches `a`, `#`, etc.                         |
| `\s`             | Whitespace                                   | Matches spaces, tabs, line breaks, etc.                                |
| `\S`             | Not whitespace                               | Matches non-space characters.                                          |
| `\w`             | Word character                               | Equivalent to `[A-Za-z0-9_]`. Matches `a`, `9`, `_`.                   |
| `\W`             | Not a word character                         | Matches any character not in `[A-Za-z0-9_]`.                           |
| `[abc]`           | Character set                                | Matches `a`, `b`, or `c`.                                              |
| `[^abc]`          | Negated character set                        | Matches anything except `a`, `b`, or `c`.                              |
| `\`<character>   | Escape special characters like `^`, `$`, etc. | Example: `\.` matches a literal period.                               |

---

## Quantifiers

| Character         | Description                                   | Example                                        |
| ----------------- | --------------------------------------------- | ---------------------------------------------- |
| `*`               | 0 or more repetitions                        | `a*` matches `", `a`, `aaaa`.                |
| `+`               | 1 or more repetitions                        | `a+` matches `a`, `aaa`, but not "".         |
| `?`               | 0 or 1 repetition                            | `a?` matches `a` or "".                      |
| `{n}`             | Exactly `n` repetitions                      | `a{3}` matches `aaa`.                         |
| `{n,}`            | At least `n` repetitions                     | `a{3,}` matches `aaa`, `aaaa`.                |
| `{n,m}`           | Between `n` and `m` repetitions              | `a{2,4}` matches `aa`, `aaa`, or `aaaa`.      |

---

## Groups

| Character         | Description                                   | Example                                        |
| ----------------- | --------------------------------------------- | ---------------------------------------------- |
| `(pattern)`       | Capturing group                              | `(ab)+` matches `abab`. Captures `ab`.        |
| `(?:pattern)`     | Non-capturing group                          | `(?:ab)+` matches `abab` but does not capture.|
| `\1`             | Backreference to first capturing group       | `(\w)\1` matches `aa` or `bb`.              |

---

## Assertions

| Character         | Description                                   | Example                                        |
| ----------------- | --------------------------------------------- | ---------------------------------------------- |
| `^`               | Start of string or line                      | `^abc` matches `abc` at the start of a string.|
| `$`               | End of string or line                        | `abc$` matches `abc` at the end of a string.  |
| `\b`             | Word boundary                                | `\bword\b` matches `word`, but not `wordy`. |
| `\B`             | Not a word boundary                          | `\Bword` matches `sword`.                    |
| `(?=pattern)`     | Positive lookahead                           | `a(?=b)` matches `a` in `ab`, but not `ac`.   |
| `(?!pattern)`     | Negative lookahead                           | `a(?!b)` matches `a` in `ac`, but not `ab`.   |

---

## Alternatives

| Character         | Description                                   | Example                                        |
| ----------------- | --------------------------------------------- | ---------------------------------------------- |
| `a|b`             | Matches `a` or `b`                           | `a|b` matches `a` or `b`.                     |

---

## Character Classes

### Individual Characters
| Example         | Description                                    |
| --------------- | --------------------------------------------- |
| `[abc]`         | Matches `a`, `b`, or `c`.                    |
| `[^abc]`        | Matches any character except `a`, `b`, or `c`|

### Ranges
| Example         | Description                                    |
| --------------- | --------------------------------------------- |
| `[a-z]`         | Matches lowercase letters.                    |
| `[A-Za-z]`      | Matches uppercase and lowercase letters.      |

### POSIX-like Classes
| Class           | Description                                   |
| --------------- | --------------------------------------------- |
| `[:alnum:]`     | Alphanumeric characters                      |
| `[:digit:]`     | Digits                                       |
| `[:space:]`     | Whitespace characters                        |

### Escape Characters
| Example         | Description                                   |
| --------------- | --------------------------------------------- |
| `\b`           | Backspace (inside character classes only)     |

---

## Useful Regex Examples

### URL Parsing
```regex
/^((\w+):)?(\/\/((\w+)?(:(\w+))?@)?([^\/\?:]+)(:(\d+))?)?(\/([^\/?#][^?#]*)?)?(\?([^#]+))?(#(\w*))?/g
```
**Test String:**
`https://user:pass@abcd.domain.com:8080/first?a=1#hash`

### Phone Number Formatting
```js
"01012345678".replace(/(\d{3})(\d{4})(\d{4})/, "$1-$2-$3");
```
**Test String:**
`01012345678`
**Output:**
`010-1234-5678`

### Add Commas to Numbers
```js
number.replace(/\B(?=(\d{3})+(?!\d))/g, ',')
```

### Validate Email
```regex
/^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/
```

### Validate Password
```regex
/^.*(?=.{8,10})(?=.*[a-zA-Z])(?=.*?[A-Z])(?=.*\d)(?=.+?[\W|_])[a-zA-Z0-9!@#$%^&*()\-_=+{}\|\\/]+$/
```

### Remove HTML Tags
```regex
/<[^>]*>/g
```
**Test String:**
`<div>Hello World</div>`
**Output:**
`Hello World`

### Validate IP Addresses
```regex
/^(?!.*\.$)((?!0\d)(1?\d?\d|25[0-5]|2[0-4]\d)(\.|$)){4}$/
```
**Test String:**
`123.255.0.1`

---

