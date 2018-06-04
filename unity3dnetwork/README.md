- [Intro](#intro)
- [Materials](#materials)
- [UNET vs PUN vs Proud](#unet-vs-pun-vs-proud)
- [unity3d](#unity3d)
  - [Server](#server)
  - [Client](#client)
  - [Cloud](#cloud)
- [Proudnet](#proudnet)
  - [Server](#server)
  - [Client](#client)
- [PUN](#pun)
  - [Server](#server)
  - [Client](#client)
  - [Cloud](#cloud)
- [Reference](#reference)

----

# Intro

- unity3d로 network를 구현하는 방법에 대해 정리해본다.

# Materials

* [barebones-masterserver](https://github.com/alvyxaz/barebones-masterserver)
  * opensource masterserver
* [Forge](https://github.com/BeardedManStudios/ForgeNetworkingRemastered)
  * opensource unity3d network framework
  * [doc](http://docs.forgepowered.com/)

# UNET vs PUN vs Proud

|  | UNET  | PUN  | Proud | Forge |
|:---:|:---:|:---:|:---:|:---:|
| Cloud Service  | o  | o  | X  | X |
| Price  |   |   |   | Free |
| Custom Server | X | O | O | O |
| Server Platform | ? | Windows | Windows, Linux | Win, Linux, Mac |
| Client Integration | NetworkIdentity, NetworkTransform, SyncVars, Command, ClientRpc | PhotonView, PhotonTransformView  |   |  |

# UNET

## Server
- NetworkManager 혹은 HLAPI로 구현하는 방법과 LLAPI로 구현하는 방법이 있다.
- LLAPI를 이용해서 dedicated server를 구현할 수 있다. Build Settings 에서 
  Target platform을 Linux로 하고 Headless mode로 빌드한다면 Linux에서 
  command line으로 실행가능하다. 현재(20170516) Headless Mode는 linux만 지원한다.
  ubuntu linux에서 잘 작동한다고 하니 참고하자. mono는 안정성과 수행성능면에서 적당할까???
- [UNET Server Hosting Tutorial](https://noobtuts.com/unity/unet-server-hosting)을 읽어보자. 
  Linux환경에서 headless server 를 실행하는 과정이 담겨 있다. [Detect Headless Mode in Unity](https://noobtuts.com/unity/detect-headless-mode)
  를 읽어보자. 다음과 같은 코드를 이용해서 현재 headless모드인지 알수 있다.
```c#
void Awake() {
    if (IsHeadless()) {
        print("headless mode detected");
        StartServer();
    }
}
```
```c#
using UnityEngine.Rendering;

// detect headless mode (which has graphicsDeviceType Null)
bool IsHeadless() {
    return SystemInfo.graphicsDeviceType == GraphicsDeviceType.Null;
}
```
- [Master Server Kit](https://www.assetstore.unity3d.com/kr/#!/content/71604)을 
  이용하면 Dedicated server를 쉽게 제작할 수 있다. 

## Client

- dedicated server가 없는 환경(p2p)이라면 여러 클라이언트중 하나가 서버가 되어야 한다.
  이것을 host라고 한다. host는 server와 local client로 구성된다. 
  그리고 host로 접근하는 client는 remote client라고 한다. 
- [TANKS! Networking Demo](https://www.assetstore.unity3d.com/kr/#!/content/46213), 
  [Tanks!!! Reference Project](https://www.assetstore.unity3d.com/kr/#!/content/80165),
  [Tanks Multiplayer](https://www.assetstore.unity3d.com/kr/#!/content/69172) 
  를 분석해보자. 
  [uMMORPG](https://www.assetstore.unity3d.com/kr/#!/content/51212)
- Object 를 Spawn하고 싶다면 NetworkManager 혹은 ClientScene.RegisterPrefab를
  이용해서 prefab을 미리 등록해야 한다. 이때 prefab의 root에 
  NetworkIdentify, NetworkBehaviour component가 attatch되어 있어야 한다.
- ClientScene.RegisterSpawnHandler 를 spawn handler를 등록할 수 있다. 
  custom poolmgr를 이용
- Player Object는 특별하다. 나의 Player Object는 isLocalPlayer가 true이다. 
  시작시 OnStartLocalPlayer가 호출된다.
- non-player object들에 대해 NetworkServer.SpawnWithClientAuthority, 
  NetworkIdentity.AssignClientAuthority를 이용해 authority를 부여할 수 있다.
  authority가 부여되면 NetworkBehaviour의 OnStartAuthority() 가 호출되고 
  hasAuthority가 true로 전환된다. authority를 부여받은 non-player object는 
  command를 보낼 수 있다. command는 server에서 실행된다. authority를 부여받을
  non-player objects는 NetworkIdentify의 LocalPlayerAuthority가 check되어야 한다.
- NetworkBehaviour class는 다음과 같은 중요한 properties를 가지고 있다.
  - isServer - true if the object is on a server (or host) and has been spawned.
  - isClient - true if the object is on a client, and was created by the server.
  - isLocalPlayer - true if the object is a player object for this client.
  - hasAuthority - true if the object is owned by the local process
- 유저의 입력을 받아 움직일때, 총을 쏠때 처리방법

## Cloud

- Unity에서 machmaking, relay server등을 cloud service한다. 
  relay server는 대한민국에서 느리다는 평가가 있다.

# Forge

Forge remastered 라는 이름으로 opensource 되었다. 아주 유용해 보인다. NCW (Network Contract Wizard) 를 이용하여 네트워크를 통해서 공유할 field, rpc method 를 적당히 추가한 후에 `ABehaviour, ANetworkObject` 스크립트를 생성한다. `ANetworkObject` instance 는 `ABehaviour` 의 member로 접근가능하다. `A` 는 `ABehaviour` 를 상속받는다. 다음과 같이 `IsServer` 를 이용하여 `A` 가 server 에서 실행되는지 client 에서 실행되는지에 따라 logic 을 달리한다.

```cs
using UnityEngine;
using BeardedManStudios.Forge.Networking.Generated;

public class A : ABehaviour {
  private void Update() {
    if (!networkObject.IsServer) {
      transform.position = networkObject.position;
      return;
    }
    transform.position += new Vector3(Input.GetAxis("Horizontal"),
    Input.GetAxis("Vertical"), 0) * Time.deltaTime * 5.0f;
    networkObject.position = transform.position;
  }

}
```

forge framework 를 활용하면 멀티세션 서버를 구현할 수 있을 것 같다. [이곳](http://docs.forgepowered.com/working-with-multiple-sockets/)을 참고해서 추론해본다. dedicated server 는 다음과 같이 port 별 server instance 를 생성하여 구현한다.

```cs
UDPServer server1 = new UDPServer(32); // Only allowing 32 connections to this UDP server
server1.Connect();

ushort myPort = 15959;
TCPServer server2 = new TCPServer(364); // Only allowing 64 connections to this TCP server
server2.Connect(myPort);

UDPServer server3 = new UDPServer(1024); // Only allowing 1024 connections to this UDP server
server3.Connect(myPort + 1);
```

client 가 특정 dedicated server 의 특정 port에 접속하기 위해서는 gate server 가 필요하다. gate server 는 특정 port 를 listen  하다가 client 가 접속하면 client 가 게임하기 위해 필요한 dedicated server ip, port 를 알려준다. 이후 client 가 dedicated server 에 접속하여 플레이 한다.

# Proudnet

## Server

- c++, c#으로 구현할 수 있다. 당연히 c++ server가 성능이 좋을 것이다. 
- c++으로 제작한다면 windows, linux에서 실행할 수 있다.
- c++ server의 경우 unity3d api를 이용할 수 없다.
- c# server의 경우 mono에서 실행한다면 UnityEngine.Dll을 이용할 수 있겠지?
- mono binary는 적당히 수정하면 dotnet core framework에서 실행 가능하다.
  - [Can I use Mono to build applications for the Compact Framework?](http://www.mono-project.com/docs/faq/technical/)
- 프로토콜전송을 위해 S2C, C2S, C2C형태의 IDL파일을 제작후 PIDL을 이용해서 구현체를 만들어낸다.
- 기본적으로 Client는 Server에 접속하는 형태이고 P2PGroup을 이용해서 C2C통신한다.
  - 중요한 것은 C2S 덜 중요한 것은 C2C 로 처리한다.
  - P2PGroup은 server에서 수행한다. client는 C2C proxy의 함수들을 이용할 뿐이다.
- punch through, relay server, dead reckoning, delayed packet queueing

## Client

- c#의 경우 native library를 사용하는 managed dll을 이용하는 방식이다. 
  core는 c++로 제작됨. c++ api의 상당부분이 managed dll에 없다. 왜일까? 구현중인가???
- PositionFollower의 경우 문서에는 내용이 없지만 ProudDotNetClient.dll에 들어 있다.

# PUN

## Server

## Client

## Cloud

# Reference

