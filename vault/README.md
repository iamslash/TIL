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

```

## API

```console
$ curl -X POST "http://local.iamslash.com:8200/v1/secret/data/hello" -H "accept: */*" -H "Content-Type: application/json" -H "X-Vault-Token: s.jzgbH3FmKEua8VqBlBgHO9iF" -d "{\"data\":{\"username\":\"iamslash\",\"password\":\"iamslash\"},\"options\":{},\"version\":0}"

$ curl -X GET "http://local.iamslash.com:8200/v1/secret/data/hello" -H "accept: */*" -H "X-Vault-Token: s.jzgbH3FmKEua8VqBlBgHO9iF"

$ curl -X DELETE "http://local.iamslash.com:8200/v1/secret/data/hello" -H "accept: */*" -H "X-Vault-Token: s.jzgbH3FmKEua8VqBlBgHO9iF"
```

