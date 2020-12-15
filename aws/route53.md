# Abstract

Amazon Route 53 is a highly available and scalable Domain Name System (DNS) web service.

# Basic

## Hosted Zone

`hello.iamslash.com` 라는 DNS name 을 만들고 싶다.

* `iamslash.com` 이라는 hosted zone 을 생성한다.
  * private 을 선택하면 등록된 VPC 에서만 DNS query 가 가능하다.
  * 해당 VPC 들을 모두 등록한다.
* `iamslash.com` hosted zone 에서 `hello.iamslash.com` RECORD 를 생성한다. RECORD type 에 따라 다양한 값을 이용할 수 있다.
  * `A` : value 에 `xxx.xxx.xxx.xxx` 와 같이 IP 를 입력한다.
  * `CNAME` : value 에 `hello.aws.com` 와 같이 다른 DNS name 을 입력한다.
  * `TXT` : value 에 text data 를 입력한다. DNS query 를 통해 text 를 조회할 수 있다.
  * `SRV` : value 에 역시 text data 를 입력한다. DNS query 를 통해 text 를 조회할 수 있다. `TXT` 와 차이???
  * 동일한 `hello.iamslash.com` 에 여러 RECORD type 을 생성할 수 있다.
* DNS query 는 cache 가 문제이다. Route 53 에서 TTL 을 짧게 설정한다고 해도 local machine 의 DNS cache 가 expire 되지 않는 이상 바뀐 값은 빨리 적용되지 않는다.


# Advanced

## How to update CNAME RECORD by command line

* [AWS CLI command to update CNAME @ gist](https://gist.github.com/tcbyrd/ffb5f596322cee976ae864f3d8061c6a)

-----

```
$ aws route53 change-resource-record-sets --hosted-zone-id [[HostedZoneID]] --change-batch file://change-cname.json
$ 
```

* `change-cname.json`

```json
{
	"Comment":"Updates CNAME to a specified value",
	"Changes":[{
		"Action":"UPSERT",
		"ResourceRecordSet":{
			"Name":[[PUBLIC DNS NAME]],
			"Type":"CNAME",
			"TTL":30, 
			"ResourceRecords":[{
				"Value":[[TARGET HOST NAME]]
			}]
		}
	}]
}
```

## How to update A RECORD by command line

* [](http://www.scalingbits.com/aws/dnsfailover/changehostnameentries)

-----

```bash
$ aws route53 change-resource-record-sets --hosted-zone-id HOSTED-ZONE-ID --change-batch file://change-a.json
```

* `chang-a.json`

```json
{
  "Comment": "Update record to reflect new IP address for a system ",
  "Changes": [
    {
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "foo.iamslash.com",
        "Type": "A",
        "TTL": 10,
        "ResourceRecords": [
          {
              "Value": "10.79.252.13"
          }
        ]
      }
    }
  ]
}
```
