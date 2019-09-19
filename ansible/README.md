# Abstract

ansible 은 infrastructure as code solution 이다. infra 를 code 로 manage 할 수 있다. palybook 을 제작해서 infra 를 구성한다.

# Materials

* [[DevOps] Windows 10에 ansible 이용하기](http://egloos.zum.com/mcchae/v/11315161)

# Install on Windows and connect from OSX

* [Windows Remote Management](https://docs.ansible.com/ansible/latest/user_guide/windows_winrm.html)

-----

ansible 은 linux 에서 SSH 를 이용하지만 Windows 에서 winrm 을 이용하여 접속한다. 다음은 winrm 을 실행하고 mac 에서 ansible 을 실행하는 예이다.

```powershell
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

PS C:\Windows\system32> $url = "https://raw.githubusercontent.com/ansible/ansible/devel/examples/scripts/ConfigureRemotingForAnsible.ps1"
PS C:\Windows\system32> $file = "$env:temp\ConfigureRemotingForAnsible.ps1"
PS C:\Windows\system32> (New-Object -TypeName System.Net.WebClient).DownloadFile($url, $file)
PS C:\Windows\system32> powershell.exe -ExecutionPolicy ByPass -File $file
Self-signed SSL certificate generated; thumbprint: 4CF0398E995729298E765A359800E320F267B3E0


wxf                 : http://schemas.xmlsoap.org/ws/2004/09/transfer
a                   : http://schemas.xmlsoap.org/ws/2004/08/addressing
w                   : http://schemas.dmtf.org/wbem/wsman/1/wsman.xsd
lang                : ko-KR
Address             : http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous
ReferenceParameters : ReferenceParameters

확인됨



PS C:\Windows\system32> Get-Service -Name winrm

Status   Name               DisplayName
------   ----               -----------
Running  winrm              Windows Remote Management (WS-Manag...


PS C:\Windows\system32> winrm enumerate winrm/config/Listener
Listener
    Address = *
    Transport = HTTP
    Port = 5985
    Hostname
    Enabled = true
    URLPrefix = wsman
    CertificateThumbprint
    ListeningOn = 10.0.75.1, 127.0.0.1, 172.17.223.161, 192.168.0.133, ::1, fe80::8442:75b5:9450:b676%3, fe80::85af:5c05:efbf:88ae%10

Listener
    Address = *
    Transport = HTTPS
    Port = 5986
    Hostname = DESKTOP-F5RAPQ0
    Enabled = true
    URLPrefix = wsman
    CertificateThumbprint = 4CF0398E995729298E765A359800E320F267B3E0
    ListeningOn = 10.0.75.1, 127.0.0.1, 172.17.223.161, 192.168.0.133, ::1, fe80::8442:75b5:9450:b676%3, fe80::85af:5c05:efbf:88ae%10
```

OSX 에서 다음과 같이 `~/my/ansible/a/requirements.txt` 를 작성한다.

```
ansible==2.7.2
pywinrm==0.3.0
```

이제 ansible 을 설치한다.

```bash
pip install -r requirements.txt
```

그리고 `hosts` 파일을 제작한다.

```yml
[win10]
192.168.0.133

[win10:vars]
ansible_user=iamslash
ansible_password=xxxx
ansible_connection=winrm
ansible_winrm_server_cert_validation=ignore

[windows:children]
win10
```

이제 ping 해보자.

```bash
ansible -i hosts win10 -m win_ping
```

이제 playbook `ping.win32.yaml` 을 제작한다.

```yml
# This playbook uses the win_ping module to test connectivity to Windows hosts
- name: Ping
  hosts: win10

  tasks:
  - name: ping
    win_ping:
```

이제 실행해보자.

```bash
ansible-playbook -i hosts ping.win32.yaml
```

이제 playbook `whoami.win32.yml` 을 제작하고 실행해 본다.

```yml
# This playbook uses the win_ping module to test connectivity to Windows hosts
- name: Whoami
  hosts: win10

  tasks:
  - name: whoami
    win_shell: whoami
```

```
ansible-playbook -i hosts -v whoami.win32.yaml
```

이제 playbook `calc.win32.yml` 을 제작하고 실행해 본다.

```yml
# This playbook tests the script module on Windows hosts
- name: Run powershell script
  hosts: win10
  gather_facts: false
  tasks:
    - name: Run calc.exe
      win_command: calc.exe
```