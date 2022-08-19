# Abstract

PKCE (Proof Key for Code Extensions) 에 대해 정리한다. "픽시" 로 발음한다.

OAUTH2.0 에 사용된다. Client, OAuth Server 가 서로 상대방이 제대로 인지 검증하기 위한 수단이다.

[OIDC](/oidc/README.md) 와 같이 이해해야 함.

# Basic

PKCE 를 포함한 OAuth Flow 는 다음과 같다. [Authorization Code With PKCE Tutorial](https://www.oauth.com/playground/client-registration.html?returnto=authorization-code-with-pkce.html) 을 실행해 보고 그 흐름을 정리해 봤다.

1. Create a Code Verifier and Challenge

Client 가 random string 을 만들어 `code_verifier` 에 저장한다. 그리고 `(base64url(sha256(code_verifier))` 를 수행하여 `code_challenge` 라고 하자.

```
 code_verifier: 77dIycYlsIMu7Hq14ulqwALdOHhLgP2eZwiNIt-LMqtNtjnc
code_challenge: MJ2vlC4jrGbcsBngD0v97nAPShXSml5HcAbuaAt4WvE
```

2. Build the authorization URL and redirect the user to the authorization server

Client 는 다시 random string 을 만들어 `state` 에 저장한다. 이 것은 CSRF 에 사용되는 것 같음. 그리고 OAuth Server 에게 HTTP Request 할 url 을 생성한다.

```
state: o2LP8ou_uLheX0VE

https://authorization-server.com/authorize?
  response_type=code
  &client_id=2LwnNURiRd4Cu-hww8lQCnw8
  &redirect_uri=https://www.oauth.com/playground/authorization-code-with-pkce.html
  &scope=photo+offline_access
  &state=o2LP8ou_uLheX0VE
  &code_challenge=MJ2vlC4jrGbcsBngD0v97nAPShXSml5HcAbuaAt4WvE
  &code_challenge_method=S256
```

OAuth Server 는 `code_challenge, code_challenge_method` 를 잘 저장해 두어 code exchange step 에서 검증을 위해 사용할 것이다.

3. Authorize

OAuth Server 가 HTTP Response 한 Login Document 를 Agent (Browser) 가 보여준다. User 는 username, password 를 입력하고 login 버튼을 누른다.

OAuth Server 가 HTTP Response 한 Approve Document 를 Agent (Browser) 가 보여준다. User 는 email, contact 등등이 공유된다라는 것을 확인하고 Approve 버튼을 누른다.

4. After the user is redirected back to the client, verify the state

OAuth Server 가 Response 한 Status 는 302 (Redirect) 이다. 전달받은 HTTP Response Header 의 location 값이 곧 redirect_url 이다. redirect_url 의 parameters 는 다음과 같다. `state` 은 이전에 Client 가 보낸 것과 같으니 redirect_url 은 믿을만 하다. `code` 는 OAuth Server 가 나중에 Client 가 요청할 HTTP Request 를 검증하기 위해 발급한 1 회용 문자열이다.

```
?state=o2LP8ou_uLheX0VE&code=tDWXFL8HEHqX9HpoA_veBj75wFcpHCDHMo9v_FAr8jln5bsa
```

5. Exchange the authorization code and code verifier for an access token

Client 가 `state` 을 검증하여 OAuth Server 가 제대로 인 것을 확인했으니 다음과 같이 token url 을 만들어 HTTP Post Request 하자.

```
POST https://authorization-server.com/token

grant_type=authorization_code
&client_id=2LwnNURiRd4Cu-hww8lQCnw8
&client_secret=vgO2shgnLqUYCMXm-p2Wn_XAQmBRXoRMZ_0wTcdQZoyq6BSf
&redirect_uri=https://www.oauth.com/playground/authorization-code-with-pkce.html
&code=tDWXFL8HEHqX9HpoA_veBj75wFcpHCDHMo9v_FAr8jln5bsa
&code_verifier=77dIycYlsIMu7Hq14ulqwALdOHhLgP2eZwiNIt-LMqtNtjnc
```

OAuth Server 는 `code` 를 읽고 예전 `code` 와 비교해 본다. 같으면 정상적인 HTTP Request 라고 판단한다.

OAuth Server 는 `code_verifier` 를 읽고 `(base64url(sha256(code_verifier))` 를 수행하여 `code_challenge` 를 만들어 낼 수 있다. 그리고 예전에 `/authorize` 를 통해 전달받은 `code_challenge` 와 비교하여 Client 가 제대로 인지 검증할 수 있다. 

6. Token Endpoint Response

```
{
  "token_type": "Bearer",
  "expires_in": 86400,
  "access_token": "H5kAEh78lPqpj24_Sfb-up-ERIMgtv43cCPi6x74_8IrCP7fkCs-D5Bx-pEdi8l6XEIqK_n8",
  "scope": "photo offline_access",
  "refresh_token": "Kfs0Zm99B-1Vq76wDC8YoWed"
}
```
