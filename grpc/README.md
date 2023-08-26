- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [grpc status code](#grpc-status-code)
  - [grpc curl client](#grpc-curl-client)

-----

# Abstract

gRPC is an open source remote procedure call (RPC) system initially developed at
Google in 2015.

# Materials

* [gRPC On HTTP/2: Engineering a robust, high performance protocol](https://www.cncf.io/blog/2018/08/31/grpc-on-http-2-engineering-a-robust-high-performance-protocol/)
* [gRPC 1 - gRPC란?](https://chacha95.github.io/2020-06-15-gRPC1/)

# Basic

## grpc status code

* [statuscodes @ github](https://github.com/grpc/grpc/blob/master/doc/statuscodes.md)

## grpc curl client

* [FullStory and gRPC](https://bionic.fullstory.com/tale-of-grpcurl/)
  * [grpcurl @ github](https://github.com/fullstorydev/grpcurl)

```bash
# fetch the repo
$ go get github.com/fullstorydev/grpcurl

# install the grpcurl command-line program
$ go install github.com/fullstorydev/grpcurl/cmd/grpcurl

# spin up a test server included in the repo
$ go install github.com/fullstorydev/grpcurl/internal/testing/cmd/testserver
$ testserver -p 9876 >/dev/null &

# and take grpcurl for a spin
$ grpcurl -plaintext localhost:9876 list
grpc.reflection.v1alpha.ServerReflection
grpc.testing.TestService

$ grpcurl -plaintext localhost:9876 list testing.TestService
grpc.testing.TestService.EmptyCall
grpc.testing.TestService.FullDuplexCall
grpc.testing.TestService.HalfDuplexCall
grpc.testing.TestService.StreamingInputCall
grpc.testing.TestService.StreamingOutputCall
grpc.testing.TestService.UnaryCall

$ grpcurl -plaintext localhost:9876 describe \
    testing.TestService.UnaryCall
grpc.testing.TestService.UnaryCall is a method:
rpc UnaryCall ( .grpc.testing.SimpleRequest ) returns
    (.grpc.testing.SimpleResponse );

# if no request data specified, an empty request is sent
$ grpcurl -emit-defaults -plaintext localhost:9876 \
    testing.TestService.UnaryCall
{
  "payload": null,
  "username": "",
  "oauthScope": ""
}

$ grpcurl -plaintext -msg-template localhost:9876 \
    describe grpc.testing.SimpleRequest
grpc.testing.SimpleRequest is a message:
message SimpleRequest {
  .grpc.testing.PayloadType response_type = 1;
  int32 response_size = 2;
  .grpc.testing.Payload payload = 3;
  bool fill_username = 4;
  bool fill_oauth_scope = 5;
  .grpc.testing.EchoStatus response_status = 7;
}

Message template:
{
  "responseType": "COMPRESSABLE",
  "responseSize": 0,
  "payload": {
    "type": "COMPRESSABLE",
    "body": ""
  },
  "fillUsername": false,
  "fillOauthScope": false,
  "responseStatus": {
    "code": 0,
    "message": ""
  }
}

$ grpcurl -emit-defaults -plaintext \
    -d '{"payload":{"body":"abcdefghijklmnopqrstuvwxyz01"}}' \
    localhost:9876 grpc.testing.TestService.UnaryCall
{
  "payload": {
    "type": "COMPRESSABLE",
    "body": "abcdefghijklmnopqrstuvwxyz01"
  },
  "username": "",
  "oauthScope": ""
}
```
