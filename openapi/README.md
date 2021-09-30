# Abstract

openapi 는 REST API 제작을 위한 specification 이다. yaml, json 으로
specification 을 정의한 후 openapi-generator CLI 로 server stubs,
client libraries 를 생성할 수 있다.

# Materials

* [What Is OpenAPI? What Is Swagger? @ swagger](https://swagger.io/docs/specification/about/)
  * [Basic Structure @ swagger](https://swagger.io/docs/specification/basic-structure/)
  * easier to understand than [OpenAPI 3.0 Specification](https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.2.md)

# Basic

## How to generate codes from the specification?

`server/template/kotlin/*.mustache` 파일을 만들고 kotlin template 을 작성한다.

`config.yml` 를 만들고 configuration 을 작성한다.

```yml
templateDir: server/template/kotlin
additionalProperties:
  reactive: "true"
  serviceInterface: "true"
  exceptionHandler: "false"
```

`spec.yml` 를 만들고 openapi 3.0 specification 을 작성한다. 여러 파일들을
import 하고 싶다면 `$ref` 를 이용하자. [yml samples](https://github.com/OpenAPITools/openapi-generator/tree/master/samples/yaml)

```yml
swagger: 2
info:
  title: Echo API
  description: Simple Rest Echo
  version: "1.0.0"
host: "localhost:8002"
schemes:
  - http
basePath: /v1
produces:
  - application/json
paths:
  /echo:
    get:
      description: "Returns the 'message' to the caller"
      operationId: "echo"
      parameters:
        #- name: X-header-param
        - name: headerParam
          in: header
          type: string
          required: false
        - name: message
          in: query
          type: string
          required: true
      responses:
        200:
          description: "Success"
          schema:
            $ref: EchoResponse
        default:
          description: "Error"
          schema:
            $ref: Error
definitions:
  EchoResponse:
    required:
      - message
    properties:
      message:
        type: string
  Error:
    properties:
      code:
        type: integer
        format: int32
      message:
        type: string
      fields:
        type: string
```

`openapi-generator` CLI 를 이용하여 code 를 생성한다.

```bash
openapi-generator generate \
-i root.yaml \
-g kotlin-spring \
-c config.yaml \
--model-package=com.iamslash
```

[openapi-generator](https://github.com/OpenAPITools/openapi-generator) 를 이용하여 server, client
에서 사용할 code 를 생성한다. [swagger codegen](https://swagger.io/docs/open-source-tools/swagger-codegen/) 을 사용하여 code 를 생성할 수도 있다.

[openapi-generator](https://github.com/OpenAPITools/openapi-generator) 는 community 에서 진행하는 open-source project 이다. [swagger codegen](https://swagger.io/docs/open-source-tools/swagger-codegen/) 은 SmartBear 라는 회사에서 주도하는 project 이다. [What is the difference between Swagger Codegen and OpenAPI Generator?](https://openapi-generator.tech/docs/faq/#what-is-the-difference-between-swagger-codegen-and-openapi-generator)
