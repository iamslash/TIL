- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
- [Advanced](#advanced)

----

# Abstract

sed for json

# Materials

 * [커맨드라인 JSON 프로세서 jq @ 44bits](https://www.44bits.io/ko/post/cli_json_processor_jq_basic_syntax)

# Basic

```bash
$ jq '.'
{"foo": "bar"}
{
  "foo": "bar"
}

$ echo 'null' | jq '.'
null

$ echo '"String"' | jq '.'
"String"

$ echo '44' | jq '.'
44    

$ echo '{"foo": "bar", "hoge": "piyo"}' | jq '.foo'
"bar"   

$ echo '{"a": {"b": {"c": "d"}}}' | jq '.a.b.c'
"d"     

$ echo '{"a": {"b": {"c": "d"}}}' | jq '.a | .b | .c'
"d"    

$ echo '[0, 11, 22, 33, 44 ,55]' | jq '.[4]'
44 

$ echo '{"data": [0, 11, 22, 33, 44 ,55]'} | jq '.data | .[4]'
44

# Loop name field in array
$ echo '[{"id":1, "name":"foo"}, {"id":2, "name": "bar"}]' | jq '.[].name'
```

# Advanced

* `$ vim z.json`
```json
{
  "data": {
      "what a burger": [1,2,3],
      "wap": [66],
      "the map": [11,20],
      "H. Incandenza": [1,1],
      "What a burger": [3,3]
  }
}
```

```bash
$ cat z.json | jq '.data | to_entries | map(select(.key | match("what a burger";"i")))'
[
  {
    "key": "what a burger",
    "value": [
      1,
      2,
      3
    ]
  },
  {
    "key": "What a burger",
    "value": [
      3,
      3
    ]
  }
]

$ cat z.json | jq '.data | to_entries | map(select(.key | match("what a burger";"i"))) | map(.value)'
[
  [
    1,
    2,
    3
  ],
  [
    3,
    3
  ]
]
```
