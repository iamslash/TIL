# Abstract 

NAT 장비를 사이에 두고 peer 끼리 통신하는 방법중 하나이다. 

# Materials

* [RFC 5128, State of Peer-to-Peer (P2P) Communication across Network Address Translators (NATs)](https://tools.ietf.org/html/rfc5128)
* [http://www.sysnet.pe.kr/2/0/1226](홀펀칭실습)
* [[UDP] 홀 펀칭 (Hole Punching)](https://elky.tistory.com/259?category=621735)
  * [[UDP] Reliable UDP](https://elky.tistory.com/258?category=621735)

# UDP Hole Punching

A (192.168.50.10:58010), NAT (175.194.21.145:60010), B (124.137.26.36:12000) 라는 peer 가 있다. A 가 B 에게  UDP 로 패킷을 전송하면 NAT 장비는 udp port (60010) 를 하나 할당하여 A와 B의 관계를 잘 저장한다. 그리고 NAT 장비는 B가 NAT (175.194.21.145:60010) 에 UDP packet 을 전송하면 A 에게 잘 전달해 준다.

C(100.100.100.100:15000) 가 NAT (175.194.21.145:60010) 에 UDP packet 을 전송하면 NAT 장비는 자신의 네트워크 테이블에 해당 정보가 없기 때문에 패킷을 버린다. 그러나 A 가 NAT 를 경유해서 C 에게 UDP packet 을 한번 전송하면 C 는 NAT 를 경유해서 A 에게 UDP 패킷을 전송 할 수 있게 된다.

* Server

```cs
UdpClient _server;
_server = new UdpClient(12000);
_server.BeginReceive(udpReceiveCallback, _server); // 비동기 데이터 수신

void udpReceiveCallback(IAsyncResult ar)
{
    try
    {
        UdpClient udpServer = ar.AsyncState as UdpClient;
        IPEndPoint remoteEndPoint = null;

        byte [] receiveBytes = udpServer.EndReceive(ar, ref remoteEndPoint);

        // 접속된 클라이언트의 IP 주소와 포트 출력
        Console.WriteLine("Receive from " + remoteEndPoint.Address.ToString() + ":" + remoteEndPoint.Port);
        udpServer.BeginReceive(udpReceiveCallback, udpServer);
    }
    catch { }
}
```

* Client

```cs
UdpClient _udpClient = new UdpClient();

private void Form1_Load(object sender, EventArgs e)
{
    IPAddress ipAddress = IPAddress.Parse("124.137.26.136");
    IPEndPoint holePunchServer = new IPEndPoint(ipAddress, 12000);

    string uid = Environment.MachineName;
    byte [] uidBytes = Encoding.UTF8.GetBytes(uid);

    _udpClient.Send(uidBytes, uidBytes.Length, holePunchServer);
}
```