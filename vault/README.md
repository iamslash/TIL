# Abstract

Credential repository 이다.

# Materials

* [vault getting-started](https://learn.hashicorp.com/vault/getting-started)

# Install with docker

* [vault @ hub.docker.com](https://hub.docker.com/_/vault)

```console
$ docker run --rm --cap-add=IPC_LOCK -e 'VAULT_DEV_ROOT_TOKEN_ID=iamslash' -p 8200:8200 -d --name=my-vault vault

$ docker logs my-vault
...
The unseal key and root token are displayed below in case you want to
seal/unseal the Vault or re-authenticate.

Unseal Key: w5JdE1QMDbGcRphrgsv1qTtQkieq2UkhDD+DwhOOhKY=
Root Token: s.D4zC6MDqW3O3EWkceC3c8VgN
...

$ docker exec -it my-vault /bin/sh
```

Open broswer 'http://localhost:8200' and login with `Root Token`. If it's first time you can start interative tutorial. It will takes just 20 mins.

# Basic

## Web-ui console

If you logged in web-ui. You can check api list on web-console with the command `> api`.

```
> api

> read secret/data/hello
```

## CLI

```console
$ docker exec -it my-vault /bin/sh

> vault login -address="http://127.0.0.1:8200" s.KNX00drphlIzcQq2avjk0POq

> vault read -address="http://127.0.0.1:8200" secret/data/iamslash

> vault auth list -address="http://127.0.0.1:8200"
Path        Type       Accessor                 Description
----        ----       --------                 -----------
approle/    approle    auth_approle_74799b27    n/a
token/      token      auth_token_df0c667f      token based credentials

> vault auth disable -address="http://127.0.0.1:8200" approle
Success! Disabled the auth method (if it existed) at: approle/

> vault auth list -address="http://127.0.0.1:8200"
Path      Type     Accessor               Description
----      ----     --------               -----------
token/    token    auth_token_df0c667f    token based credentials

> vault auth enable -address="http://127.0.0.1:8200" approle
Success! Disabled the auth method (if it existed) at: approle/

> vault auth list -address="http://127.0.0.1:8200"
Path        Type       Accessor                 Description
----        ----       --------                 -----------
approle/    approle    auth_approle_af4185df    n/a
token/      token      auth_token_df0c667f      token based 
credentials

> vault write -address="http://127.0.0.1:8200" -f auth/approle/role/my-role

> vault read -address="http://127.0.0.1:8200" auth/approle/role/my-role

> vault write -address="http://127.0.0.1:8200" auth/approle/role/my-role \
    secret_id_ttl=10m \
    token_num_uses=10 \
    token_ttl=20m \
    token_max_ttl=30m \
    secret_id_num_uses=40

> vault read -address="http://127.0.0.1:8200" auth/approle/role/my-role/role-id

> vault write -address="http://127.0.0.1:8200" -f auth/approle/role/my-role/secret-id
Key                   Value
---                   -----
secret_id             3b6a24d5-f3e6-8fa0-a545-97fdfad36427
secret_id_accessor    4ad88efe-450e-1b81-1761-8a628d2f1046

> vault secrets list -address=http://127.0.0.1:8200 -detailed
```

## API

```console
$ curl -X POST "http://local.iamslash.com:8200/v1/secret/data/hello" -H "accept: */*" -H "Content-Type: application/json" -H "X-Vault-Token: s.jzgbH3FmKEua8VqBlBgHO9iF" -d "{\"data\":{\"username\":\"iamslash\",\"password\":\"iamslash\"},\"options\":{},\"version\":0}"

$ curl -X GET "http://local.iamslash.com:8200/v1/secret/data/hello" -H "accept: */*" -H "X-Vault-Token: s.jzgbH3FmKEua8VqBlBgHO9iF"

$ curl -X DELETE "http://local.iamslash.com:8200/v1/secret/data/hello" -H "accept: */*" -H "X-Vault-Token: s.jzgbH3FmKEua8VqBlBgHO9iF"
```

