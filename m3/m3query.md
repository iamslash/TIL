# Abstract

service that coordinates reads and writes between upstream systems, such as Prometheus, and M3DB.

# APIs

* [M3 Query API](https://m3db.io/docs/m3query/api/query/)

## `/api/v1/query_range GET`

```bash
$ curl 'http://localhost:7201/api/v1/query_range?query=abs(http_requests_total)&start=1530220860&end=1530220900&step=15s'
{
  "status": "success",
  "data": {
    "resultType": "matrix",
    "result": [
      {
        "metric": {
          "code": "200",
          "handler": "graph",
          "method": "get"
        },
        "values": [
          [
            1530220860,
            "6"
          ],
          [
            1530220875,
            "6"
          ],
          [
            1530220890,
            "6"
          ]
        ]
      },
      {
        "metric": {
          "code": "200",
          "handler": "label_values",
          "method": "get"
        },
        "values": [
          [
            1530220860,
            "6"
          ],
          [
            1530220875,
            "6"
          ],
          [
            1530220890,
            "6"
          ]
        ]
      }
    ]
  }
}
```


