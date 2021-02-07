- [Abstract](#abstract)
- [Materials](#materials)
- [Stateless Authentication](#stateless-authentication)
- [Basic Structure](#basic-structure)

----

# Abstract

JWT 에 정리한다.

# Materials

* [JWT 연습 프로젝트 @ github](https://github.com/appkr/jwt-scratchpad)
* [[JWT] JSON Web Token 소개 및 구조](https://velopert.com/2389)

# Stateless Authentication

**HTTP Basic Authentication**

username, password 를 base64 encoding 하여 `Authorization: Basic am9obkBleGFtcGxlLmNvbTpwYXNzd29yZA==` 와 같이 `Authorization` header 에 담아서 보낸다.

`Authorization` header 를 탈취하면 누구나 열람할 수 있다.

**OATUH**

Authentication 을 위해 Client 는 `ClientId, ClientSecret` 이 필요하다. `ClientId, ClientSecret` 을 발급절차를 포함해서 복잡하다.

**JWT**

JWT 안에 필요한 정보들이 저장되어 있다. HTTPS 로 주고 받으면 탈취당할 염려는 없다. 또한 `Head, Payload, Signature` 로 구성되어 있는데 `Signature` 를 이용하여 변조여부를 판단할 수 있다.

# Basic Structure

* [jwt debugger](https://jwt.io/)

----

다음과 같이 `<HEADER>.<PAYLOAD>.<SIGNATURE>` 로 성되어 있다.

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
```

**HEADER**

```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```

**PAYLOAD**

```json
{
  "sub": "1234567890",
  "name": "John Doe",
  "iat": 1516239022
}
```

**SIGNATURE**

```json
HMACSHA256(
  base64UrlEncode(header) + "." +
  base64UrlEncode(payload),
  your-256-bit-secret
) secret base64 encoded
```
