# Abstract

LDAP 는 DAP 를 간소화한 프로토콜이다. 트리구조를 갖는 오브젝트 데이터를 전송하는데 사용된다. 대표적인 구현체로 MS 의 Active Directory 와 [OpenLDAP](http://www.openldap.org/) 가 있다. 대표적인 클라이언트 구현체로 [Apache Directory Studio](https://directory.apache.org/studio/) 가 있다. [이곳](https://directory.apache.org/studio/users-guide/2.0.0.v20180908-M14/ldap_browser/tools_search_dialog.html) 를 참고하면 [Apache Directory Studio](https://directory.apache.org/studio/) 를 이용한 검색방법을 알 수 있다.

macOS 에 ldapsearch 가 기본적으로 설치되어 있고 ldap 조회를 실행할 수 있다. password 는 `-W` 옵션을 주고 매번 입력하거나 `passwd.txt` 에 저장하여 매번 입력하지 않을 수 있다.

`/etc/openldap/ldap.conf` 는 openldap client 의 설정파일이다. `/etc/openldap/slapd.conf` 는 openldap server 의 설정파일이다. 설정파일 사용법은 나중에 정리한다.

# Materials

* [Examples of using ldapsearch](https://www.ibm.com/support/knowledgecenter/en/SSKTMJ_9.0.1/admin/conf_examplesofusingldapsearch_t.html)
* [How to do ldapsearch Example @ youtube](https://www.youtube.com/watch?v=sFGq7k31B-I)

# openldap client 

```bash
# -H : AD 서버 주소
# -x : 단순 인증
# -D : AD 로그인 정보
# -y : 패스워드 파일 이름
# -b : 검색이 시작되는 도메인 이름
# cn 이 David 로 시작하는 object 검색해서 cn 를 리턴
$ ldapsearch -H ldap://xxx.xxx.xxx.xxx -x -D iamslash@com.iamslash.net -y passwd.txt -b "DC=corp,DC=iamslash,DC=net" "(cn=David*)" cn
```