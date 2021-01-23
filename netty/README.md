# Abstract

an asynchronous event-driven network application framework.

# Materials

* [Introduction to Netty @ baeldung](https://www.baeldung.com/netty)

# Basic

## Concepts

**Channel**

It represents an open connection which is capable of IO operations such as reading and writing.

**Future**

Every IO operation on a Channel in Netty is non-blocking. Netty has its own **ChannelFuture** interface. We can pass a callback to **ChannelFuture** which will be called upon operation completion.

**Events and Handlers**

There 2 events including inbound, outbound events.

Inbound Events

* Channel activation and deactivation
* Read operation events
* Exception events
* User events

Outbound Events

* opening/closing a connection 
* writing/flushing data.

Channel event handlers are **ChannelHandler** and its ancestors **ChannelOutboundHandler** and **ChannelInboundHandler**.

**Encoders and Decoders**

Netty introduces special extensions of the **ChannelInboundHandler** for decoders which are capable of decoding incoming data. The base class of most **decoders** is **ByteToMessageDecoder**.

Netty has extensions of the **ChannelOutboundHandler** called encoders. **MessageToByteEncoder** is the base for most **encoder** implementations.

## Examples

* [netty tutorial @ github](https://github.com/eugenp/tutorials/tree/master/libraries-server/src/main/java/com/baeldung/netty)
