# intro

- unity3d로 network를 구현하는 방법에 대해 정리해본다.

# Library

|  | UNET  | PUN  | Proud  |
|:---:|:---:|:---:|:---:|
| Cloud Service  | o  | o  | X  |
| Price  |   |   |   |
| Custom Server | X | O | O |
| Server Platform | ? | Windows | Windows, Linux |
| Client Integration | NetworkIdentity, NetworkTransform, SyncVars, Command, ClientRpc | PhotonView, PhotonTransformView  |   |

# unity3d

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
- [Forge Networking Headless Linux Server](https://www.youtube.com/watch?v=qxm-071uLuE)을 시청하자.
  Forge Networking library를 이용하여 linux에서 headless server를 실행할 수 있다. 
  Forge Networking library는 유료다.

## Client

- dedicated server가 없는 환경(p2p)이라면 여러 클라이언트중 하나가 서버가 되어야 한다.
  이것을 host라고 한다. host는 server와 local client로 구성된다. 
  그리고 host로 접근하는 client는 remote client라고 한다. 
- [TANKS! Networking Demo](https://www.assetstore.unity3d.com/kr/#!/content/46213), 
  [Tanks!!! Reference Project](https://www.assetstore.unity3d.com/kr/#!/content/80165),
  [Tanks Multiplayer](https://www.assetstore.unity3d.com/kr/#!/content/69172) 
  를 분석해보자. 

## Cloud

- Unity에서 machmaking, relay server등을 cloud service한다. 
  relay server는 대한민국에서 느리다는 평가가 있다.

# Conclustion

- 
