- [Basic](#basic)
  - [**POSIX 정규 표현식 (POSIX ERE, Extended Regular Expressions)**](#posix-정규-표현식-posix-ere-extended-regular-expressions)
  - [**Perl 호환 정규 표현식 (PCRE, Perl-Compatible Regular Expressions)**](#perl-호환-정규-표현식-pcre-perl-compatible-regular-expressions)
  - [**ECMAScript 정규 표현식 (JavaScript)**](#ecmascript-정규-표현식-javascript)
  - [**.NET 정규 표현식 (C#)**](#net-정규-표현식-c)
  - [**Python 정규 표현식 (`re` 모듈)**](#python-정규-표현식-re-모듈)
  - [**비교 요약**](#비교-요약)

---

## Materials
* [ECMAScript Regular Expressions Pattern Syntax](http://www.cplusplus.com/reference/regex/ECMAScript/)
* [Regex101](https://regex101.com/): Test PCRE, ECMAScript, Python, Golang regex.

---

# Basic

##  **POSIX 정규 표현식 (POSIX ERE, Extended Regular Expressions)**
**특징**  
- 비교적 **오래된 정규 표현식 표준**으로, Unix 계열 시스템에서 사용됨.
- **MySQL, awk, grep, sed, POSIX C 라이브러리**에서 사용됨.
- `\w`, `\d`, `\s` 등의 **단축 문자 클래스**를 지원하지 않음.
- `|`, `+`, `?`, `()` 등의 연산자를 지원하는 **ERE(Extended Regular Expression)** 버전이 있음.
- **비교적 속도가 빠르고 가로운** 반면, 유용성이 복잡함.

**주요 문법**
| 기능        | POSIX ERE 표현식 |
|------------|----------------|
| 숫자        | `[0-9]`       |
| 문자        | `[a-zA-Z]`    |
| 단어 문자    | `[a-zA-Z0-9_]` (PCRE의 `\w` 대처) |
| 공백 문자    | `[[:space:]]` (PCRE의 `\s` 대처) |
| 이메일 예제 | `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}` |

**사용 예제 (MySQL, grep)**
```sql
SELECT * FROM users WHERE email REGEXP '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$';
```
```sh
echo "hello123" | grep -E '[[:digit:]]+'
```

---

##  **Perl 호환 정규 표현식 (PCRE, Perl-Compatible Regular Expressions)**

**특징**  
- Perl 언어에서 파제된 정규 표현식 문법으로, **매우 강력한 기능**을 제공.
- **PostgreSQL, PHP, Python(re 모듈), JavaScript, .NET, Java, Ruby** 등에서 사용됨.
- `\w`, `\d`, `\s`, `\b` 같은 단축 문자를 지원.
- **Lookahead(전방 탐색) 및 Lookbehind(후방 탐색)** 같은 고급 기능을 지원.

**주요 문법**
| 기능        | PCRE 표현식 |
|------------|-------------|
| 숫자        | `\d`       |
| 문자        | `[a-zA-Z]` |
| 단어 문자    | `\w` (`[a-zA-Z0-9_]` 대처) |
| 공백 문자    | `\s` |
| Lookahead  | `(?=pattern)` |
| Lookbehind | `(?<=pattern)` |
| 이메일 예제 | `^\w+@\w+\.\w{2,}$` |

**사용 예제 (PostgreSQL, Python, JavaScript)**

```sql
SELECT * FROM users WHERE email ~ '^\w+@\w+\.\w{2,}$';
```

```python
import re
re.match(r'^\w+@\w+\.\w{2,}$', 'user@example.com')
```

```javascript
const regex = /^\w+@\w+\.\w{2,}$/;
console.log(regex.test("user@example.com"));
```

---

## **ECMAScript 정규 표현식 (JavaScript)**

**특징**  
- **JavaScript에서 사용되는 정규 표현식**으로, PCRE와 유사하지만 일부 차이가 존재.
- `new RegExp()` 또는 `/pattern/` 형태로 사용.
- Lookbehind `(?<=...)` 지원이 부적합한 경우가 있음(최신 브라우저에서 지원).

**사용 예제 (JavaScript)**

```javascript
const regex = /^\w+@\w+\.\w{2,}$/;
console.log(regex.test("user@example.com")); // true
```

---

##  **.NET 정규 표현식 (C#)**

**특징**  

- **PCRE와 유사하지만 일부 기능이 다름.**
- C#에서 `Regex` 클래스를 통해 사용.

**사용 예제 (C#)**

```csharp
using System.Text.RegularExpressions;

string pattern = @"^\w+@\w+\.\w{2,}$";
bool isMatch = Regex.IsMatch("user@example.com", pattern);
Console.WriteLine(isMatch); // true
```

---

##  **Python 정규 표현식 (`re` 모듈)**

**특징**  

- **PCRE 기반이지만 일부 차이가 있음.**
- `re` 모듈에서 `re.compile()`, `re.match()`, `re.search()` 등을 제공.

**사용 예제 (Python)**
```python
import re
pattern = r'^\w+@\w+\.\w{2,}$'
match = re.match(pattern, 'user@example.com')
print(bool(match))  # True
```

---

## **비교 요약**

| 정규식 엔진 | 사용 환경 | 특징 |
|------------|-------------|------|
| **POSIX ERE** | MySQL, grep, sed | 기본적인 기능만 제공, `\w` 미지원 |
| **PCRE** | PostgreSQL, PHP, Python, JavaScript | 가장 강력한 기능, `\w`, `\d`, `\s` 지원 |
| **ECMAScript** | JavaScript | PCRE 기반이지만 일부 기능 제한 |
| **.NET Regex** | C# (Regex 클래스) | PCRE 유사, C# 특화 기능 추가 |
| **Python `re`** | Python | PCRE 기반이지만 일부 차이 |
| **RE2** | Go (regexp 패키지) | 메모리 폭발 방지, Lookbehind 미지원 |

---
