# Abstract

GoMock is a mocking framework for the Go programming language.

# Materials

* [gomock](https://github.com/golang/mock)

# Basic

WIP...

# Advanced

## Make AWS Cloudformation mock.go

* [aws-nuke @ github](https://github.com/rebuy-de/aws-nuke/blob/master/resources/cloudformation-stack_test.go)

```bash
#!/bin/sh
go run github.com/golang/mock/mockgen -source $(go list -m -f "{{.Dir}}" "github.com/aws/aws-sdk-go")/service/cloudformation/cloudformationiface/interface.go -destination mocks/mock_cloudformationiface/mock.go
```
