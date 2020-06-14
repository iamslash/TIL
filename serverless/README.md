# Abstract

Develop, deploy, monitor and secure serverless applications on any cloud.

# Materials

* [serverless](https://www.serverless.com/)
* [Serverless에서 Python Lambda 생성 및 배포하기](https://velog.io/@_gyullbb/Serverless%EC%97%90%EC%84%9C-Python-Lambda-%EC%83%9D%EC%84%B1-%EB%B0%8F-%EB%B0%B0%ED%8F%AC%ED%95%98%EA%B8%B0-5)
* [How to Handle your Python packaging in Lambda with Serverless plugins](https://www.serverless.com/blog/serverless-python-packaging/)


# Install serverless

```console
$ npm install -g serverless
```

# Basic

## Simple python3 AWS Lambda Function

### Initial Setup

```console
$ npm install -g serverless
```

### Creating your service locally

Create template files.

```console
$ serverless create \
  --template aws-python3 \
  --name numpy-test \
  --path numpy-test
```

Activate virtual environment.

```console
$ cd numpy-test
$ virtualenv venv --python=python3
Running virtualenv with interpreter /usr/local/bin/python3
Using base prefix '/usr/local/Cellar/python3/3.6.1/Frameworks/Python.framework/Versions/3.6'
New python executable in /Users/username/scratch/numpy-test/venv/bin/python3.6
Also creating executable in /Users/username/scratch/numpy-test/venv/bin/python
Installing setuptools, pip, wheel...done.
$ source venv/bin/activate
(venv) $
```

Update `handler.py`.

```console
$ vim handler.py
```

```py
# handler.py

import numpy as np

def main(event, context):
    a = np.arange(15).reshape(3, 5)

    print("Your numpy array:")
    print(a)

if __name__ == "__main__":
    main('', '')
```

Try to execute it.

```console
(venv) $ python handler.py
Traceback (most recent call last):
  File "handler.py", line 1, in <module>
    import numpy as np
ImportError: No module named numpy
```

Install `numpy`.

```console
(venv) $ pip install numpy
Collecting numpy
  Downloading numpy-1.13.1-cp36-cp36m-macosx_10_6_intel.macosx_10_9_intel.macosx_10_9_x86_64.macosx_10_10_intel.macosx_10_10_x86_64.whl (4.5MB)
    100% |████████████████████████████████| 4.6MB 305kB/s
Installing collected packages: numpy
Successfully installed numpy-1.13.1
(venv) $ pip freeze > requirements.txt
(venv) $ cat requirements.txt
numpy==1.13.1
```

Test `handler.py` in local.

```console
(venv) $ python handler.py
Your numpy array:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
```

### Deploying your service

Update `serverless.yml`.

```console
$ vim serverless.yml
```

```yml
# serverless.yml

service: numpy-test

provider:
  name: aws
  runtime: python3.6

functions:
  numpy:
    handler: handler.main
```

Install `serverless-python-requirements` plugin.

```console
(venv) $ npm init
This utility will walk you through creating a package.json file.

...Truncated...

Is this ok? (yes) yes

(venv) $ npm install --save serverless-python-requirements
```

Update `serverless.yml`.

```console
$ vim serverless.yml
```

```yml
# serverless.yml

plugins:
  - serverless-python-requirements

custom:
  pythonRequirements:
    dockerizePip: non-linux
```

Deploy serverless application.

```console
(venv) $ serverless deploy
Serverless: Parsing Python requirements.txt
Serverless: Installing required Python packages for runtime python3.6...
Serverless: Docker Image: lambci/lambda:build-python3.6
Serverless: Linking required Python packages...

... Truncated ...

Serverless: Stack update finished...
Service Information
service: numpy-test
stage: dev
region: us-east-1
api keys:
  None
endpoints:
  None
functions:
  numpy: numpy-test-dev-numpy
```

Invoke serverless application.

```console
(venv) $ serverless invoke -f numpy --log
--------------------------------------------------------------------
START RequestId: b32af7a8-52fb-4145-9e85-5985a0f64fe4 Version: $LATEST
Your numpy array:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
END RequestId: b32af7a8-52fb-4145-9e85-5985a0f64fe4
REPORT RequestId: b32af7a8-52fb-4145-9e85-5985a0f64fe4	Duration: 0.52 ms	Billed Duration: 100 ms 	Memory Size: 1024 MB	Max Memory Used: 37 MB
```

