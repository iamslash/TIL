- [Materials](#materials)
- [Special pattern characters](#special-pattern-characters)
- [Quantifiers](#quantifiers)
- [Groups](#groups)
- [Assertions](#assertions)
- [Alternatives](#alternatives)
- [Character classes](#character-classes)
  - [Individual characters](#individual-characters)
  - [Ranges](#ranges)
  - [POSIX-like classes](#posix-like-classes)
  - [Escape characters](#escape-characters)
- [Useful Regex](#useful-regex)

-----

# Materials

* [ECMAScript regular expressions pattern syntax](http://www.cplusplus.com/reference/regex/ECMAScript/)
* [regular expressions 101](https://regex101.com/)
  * Test PCRE, ECMAScript, Python, Golang regex.

# Special pattern characters

| characters          | description                                   | example                                                                                                                                   |
| ------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `.`                 | not newline such as LF, CR, LS, PS            | `.+` matches '<u>a b c</u>'                                                                                                               |
| `\t`                | horizontal tab (HT) same as `\u0009`          |                                                                                                                                           |
| `\n`                | newline (LF) same as`\000A`                   |                                                                                                                                           |
| `\v`                | vertical tab (VT) same as `\u000B`            |                                                                                                                                           |
| `\f`                | form feed (FF) same as `\uu000C`              |                                                                                                                                           |
| `\r`                | carriage return (CR) same as `\u000D`         |                                                                                                                                           |
| `\c`<i>letter</i>   | control code                                  | `\ca` matches the control sequence `CTRL+a`                                                                                               |
| `\x`<i>hh</i>       | ASCII character                               | `\x4c` is the same as `L`, or `\x23` the same as `#`                                                                                      |
| `\u`<i>hhhh</i>     | unicode character                             |                                                                                                                                           |
| `\0`                | null same as \u0000                           |                                                                                                                                           |
| `\int`              | backreference                                 | `([a-z]+)([A-Z]+)` matches "<u>abcdb</u><u>ABCDE</u>abcdb". `/1` means first group "<u>abcdb</u>". `/2` emans second group "<u>ABCDE</u>" |
| `\d`                | digit                                         | same as `[[:digit:]]`                                                                                                                     |
| `\D`                | not digit                                     | same as `[^[:digit:]]`                                                                                                                    |
| `\s`                | whitespace                                    | same as `[[:space:]]`                                                                                                                     |
| `\S`                | not whitespace                                | same as `[^[:space:]]`                                                                                                                    |
| `\w`                | word. an alphanumeric or underscore character | same as `[_[:alnum:]]`                                                                                                                    |
| `\W`                | not word.            | same as `[^_[:alnum:]]`   |
| `\`<i>character</i> | escaped character                             | Needed for ` ^ $ \ . * + ? ( ) [ ] { } |`                                                                                                 |
| `[class]`           | character class                               | `[abc]+` matches '<u>a</u> <u>bb</u> <u>ccc</u>'                                                                                          |
| `[^class]`          | negated character class                       | `[^abc]+` matches '<u>Anything </u>b<u>ut </u>abc<u>.</u>'                                                                                |

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
| `(?:subpattern)` | passive group | `([a-z]+)([A-Z]+)` matches "<u>abcdb</u><u>ABCDE</u>abcdbe". `\1` means the first group "<u>abcdb</u>". `\2` means the second group "<u>ABCDE</u>". `(?:[a-z]+)([A-Z]+)` matches "<u>abcdb</u><u>ABCDE</u>abcdbe". But won't create a  capture group "<u>abcdb</u>". `\1` means the first group "<u>ABCDE</u>" |

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

A character class defines a category of characters in square brackets `[]`

## Individual characters

* `[abc]+` matches '<u>a</u> <u>bb</u> <u>ccc</u>'.
* `[^abc]+` matches '<u>Anything </u>b<u>ut </u>abc<u>.</u>'.

## Ranges

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

* `\b` in character class is interpreted as a backspace character `\u0008` instead of a word boundary.

# Useful Regex

* [자주 사용하는 정규식 패턴](https://uznam8x.tistory.com/m/entry/%EC%9E%90%EC%A3%BC-%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94-%EC%A0%95%EA%B7%9C%EC%8B%9D-%ED%8C%A8%ED%84%B4)

----

* URL Parse
  * Regex 

    `/^((\w+):)?(\/\/((\w+)?(:(\w+))?@)?([^\/\?:]+)(:(\d+))?)?(\/?([^\/\?#][^\?#]*)?)?(\?([^#]+))?(#(\w*))?/g`
  * test string
    `https://user:pass@abcd.domain.com:8080/first?a=1#hash`

* Phone Number
  * Regex
    ```js
    "01012345678".replace(/(\d{3})(\d{4})(\d)/, "$1-$2-$3");
    ```
  * test string
    `01012345678`
  * output
    `010-1234-5678`
* Input `,` every 3 characters
  * Regex
    ```js
    number.replace(/\B(?=(\d{3})+(?!\d))/g, ',')
    ```

* HTML attributes
  * Regex
    ```js
    const regex = /(\S+)=["']?((?:.(?!["']?\s+(?:\S+)=|[>"']))*.)["']?/g;
    const str = `<div><a href="http://www.domain.com" target="blank"><img src="https://www.image.com/image.jpg" alt="a" /></a></div>`;
    let m;

    while ((m = regex.exec(str)) !== null) {
        m.forEach((match, index) => {
            console.log(`${index}: ${match}`);
        });
    }    
    ```
  * test string
    ```html
    <div><a href="http://www.domain.com" target="_blank"><img src="https://www.image.com/image.jpg" alt="a" /></a></div>
    ```
* Date format
  * Regex
    ```js
    "20200110".replace(/(\d{4})(\d{1,2})(\d{1,2})/, ($f, $1, $2, $3) => type.replace('YYYY', $1).replace('MM', $2).replace('DD', $3));
    ```
* HTML Encode
  * Regex
    ```js
    "<div>Hello World</div>".replace(/[\u00A0-\u9999<>\&]/gim, (v) => '&#' + v.charCodeAt(0) + ';');
    ```
  * test string
    `<div>Hello World</div>`
  * output
    `&#60;div&#62;Hello World&#60;/div&#62;`
* Remove HTML Tag
  * Updating...

* Validate Emails
  * Regex
    ```js
    const regex = /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/g;
    const str = "iamslash@gmail.com";
    regex.test(str);    
    ```
* Validate Passwords
  * Regex
    ```js
    const regex = /^.*(?=.{8,10})(?=.*[a-zA-Z])(?=.*?[A-Z])(?=.*\d)(?=.+?[\W|_])[a-zA-Z0-9!@#$%^&*()-_+={}\|\\\/]+$/g;
    const str = "aAzZ1!a_#";
    regex.test(str);    
    ```
* Validate IPs
  * Regex
    ```js
    const regex = /^(?!.*\.$)((?!0\d)(1?\d?\d|25[0-5]|2[0-4]\d)(\.|$)){4}$/g;
    const str = "123.255.0.1";
    regex.test(str);    
    ```
* Validate MAC addresses
  * Regex
    ```js
    const regex = /^([0-9a-fA-F]{2}[:.-]?){5}[0-9a-fA-F]{2}$/g;
    const str = "AA:BB:8c:DD:12:FF";
    regex.test(str);    
    ```
  * test string
    `AA:BB:8c:DD:12:FF`
* Specific Characters
  * Regex
    ```js
    /^[0-9]+$/g.test(1234);        // only numbers
    /^[a-zA-Z]+$/g.test("abcd");   // only alphas
    /^[가-힣]+$/g.test("가나다라"); // only korean
    ```
* Validate Internation Phone Numbers
  * Regex
    ```js
    const regex = /(\+|00)(297|93|244|1264|358|355|376|971|54|374|1684|1268|61|43|994|257|32|229|226|880|359|973|1242|387|590|375|501|1441|591|55|1246|673|975|267|236|1|61|41|56|86|225|237|243|242|682|57|269|238|506|53|5999|61|1345|357|420|49|253|1767|45|1809|1829|1849|213|593|20|291|212|34|372|251|358|679|500|33|298|691|241|44|995|44|233|350|224|590|220|245|240|30|1473|299|502|594|1671|592|852|504|385|509|36|62|44|91|246|353|98|964|354|972|39|1876|44|962|81|76|77|254|996|855|686|1869|82|383|965|856|961|231|218|1758|423|94|266|370|352|371|853|590|212|377|373|261|960|52|692|389|223|356|95|382|976|1670|258|222|1664|596|230|265|60|262|264|687|227|672|234|505|683|31|47|977|674|64|968|92|507|64|51|63|680|675|48|1787|1939|850|351|595|970|689|974|262|40|7|250|966|249|221|65|500|4779|677|232|503|378|252|508|381|211|239|597|421|386|46|268|1721|248|963|1649|235|228|66|992|690|993|670|676|1868|216|90|688|886|255|256|380|598|1|998|3906698|379|1784|58|1284|1340|84|678|681|685|967|27|260|263)(9[976]\d|8[987530]\d|6[987]\d|5[90]\d|42\d|3[875]\d|2[98654321]\d|9[8543210]|8[6421]|6[6543210]|5[87654321]|4[987654310]|3[9643210]|2[70]|7|1)\d{4,20}$/g
    regex.test("+821012345678");
    ```
  * test string
    `+821012345678`
* Upper First Character
  * Regex
    ```js
    "hello world".replace(/\b[a-z]/g, (v) => v.toUpperCase())
    ```
  * test string
    `hello world`
  * result string
    `Hello World` 
