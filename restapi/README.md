- [Materials](#materials)
- [Basic](#basic)
  - [Make sure to use nouns in endpoint paths](#make-sure-to-use-nouns-in-endpoint-paths)
  - [JSON as the main format for sending and receiving data](#json-as-the-main-format-for-sending-and-receiving-data)
  - [Use a set of predictable HTTP status codes](#use-a-set-of-predictable-http-status-codes)
  - [Return standardized messages](#return-standardized-messages)
  - [Use pagination, filtering and sorting when fetching collections of records](#use-pagination-filtering-and-sorting-when-fetching-collections-of-records)
  - [PATCH instead of PUT](#patch-instead-of-put)
  - [Provide extended response options](#provide-extended-response-options)
  - [Endpoint Responsibility](#endpoint-responsibility)
  - [Provide Accurate API Documentation](#provide-accurate-api-documentation)
  - [Use SSL for Security and configure CORS](#use-ssl-for-security-and-configure-cors)
  - [Version the API](#version-the-api)
  - [Cache data to improve performance](#cache-data-to-improve-performance)
  - [Use standard UTC dates](#use-standard-utc-dates)
  - [A health check endpoint](#a-health-check-endpoint)
  - [Accept API key authentication](#accept-api-key-authentication)

----

# Materials

* [15 fundamental tips on REST API design | medium](https://medium.com/@liams_o/15-fundamental-tips-on-rest-api-design-9a05bcd42920)
* [EP53: Design effective and safe APIs](https://blog.bytebytego.com/p/ep53-design-effective-and-safe-apis)
* [Mastering the Art of API Design](https://blog.bytebytego.com/p/api-design?utm_source=substack&utm_medium=email)
* [Design Effective and Secure REST APIs](https://blog.bytebytego.com/p/design-effective-and-secure-rest)

# Basic

## Make sure to use nouns in endpoint paths

동사는 `GET, POST, PATCH, PUT, DELETE` 로 충분하다. 명사만 사용하자.

## JSON as the main format for sending and receiving data

JSON (JavaScript Object Notation) is enough.

## Use a set of predictable HTTP status codes

API 의 결과는 HTTP status code 규격에 충실하자.

```
200 for general success
201 for successful creation
400 for bad requests from the client like invalid parameters
401 for unauthorized requests
403 for missing permissions onto the resources
404 for missing resources
429 for too many requests
5xx for internal errors (these should be avoided as much as possible)
```

## Return standardized messages

Response body 는 규격을 정해야 한다.

```json
{
   "data": [ 
     {
       "bookId": 1,
       "name": "The Republic"
     },
     {
       "bookId": 2,
       "name": "Animal Farm"
     }
   ],
   "totalDocs": 200,
   "nextPageId": 3
}
```

다음과 같이 error code, message 를 추가해도 좋다. `message` 는 user 에게 보여줄 내용이다.

```js
{
  "code": "book/not_found",
  "message": "A book with the ID 6 could not be found"
}
```

## Use pagination, filtering and sorting when fetching collections of records

> [Users | GitHub API](https://docs.github.com/en/rest/users/users?apiVersion=2022-11-28)

```
GET /users

Query parameters

since integer
A user ID. Only return users with an ID greater than this ID.

per_page integer (default: 30)
The number of results per page (max 100).
```

## PATCH instead of PUT

## Provide extended response options

```js
GET /books/:id
{
   "bookId": 1,
   "name": "The Republic"
}
GET /books/:id?extended=true
{
   "bookId": 1,
   "name": "The Republic"
   "tags": ["philosophy", "history", "Greece"],
   "author": {
      "id": 1,
      "name": "Plato"
   }
}
```

## Endpoint Responsibility

Single Responsibility Principle

## Provide Accurate API Documentation

## Use SSL for Security and configure CORS

## Version the API

```
1. Adding a new header "x-version=v2"
2. Having a query parameter "?apiVersion=2"
3. Making the version part of the URL: "/v2/books/:id"
```

## Cache data to improve performance

##  Use standard UTC dates

```json
{
    "createdAt": "2022-03-08T19:15:08Z"
}
```

## A health check endpoint

## Accept API key authentication

HTTP header (such as `Api-Key` or `X-Api-Key`)
