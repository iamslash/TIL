# Abstract

AWS 에서 제공하는 IAC (Infrastructure As a Code) SDK 이다. "csharp", "fsharp", "java", "javascript", "python", "typescript" 등을 지원한다. 

# Materials

* [코드로 인프라 관리: AWS CDK](https://musma.github.io/2019/11/28/about-aws-cdk.html)
* [Getting started with the AWS CDK](https://docs.aws.amazon.com/cdk/latest/guide/getting_started.html)

# Hello World

```bash
# Install aws-cdk
$ npm install -g aws-cdk

$ mkdir a
$ cd a

# Init project
$ cdk init --language typescript

$ tree -L 1 .
.
├── README.md
├── bin
├── cdk.json
├── jest.config.js
├── lib
├── node_modules
├── package-lock.json
├── package.json
├── test
└── tsconfig.json

$ cat lib/a-stack.ts
```

* lib/a-stack.ts

```ts
import * as cdk from '@aws-cdk/core';

export class AStack extends cdk.Stack {
  constructor(scope: cdk.Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // The code that defines your stack goes here
  }
}
```

```bash
$ vim lib/a-stack.ts
```

* lib/a-stack.ts

```ts
import {
  SecurityGroup,
  Vpc,
  Port,
  AmazonLinuxGeneration,
  AmazonLinuxImage,
  InstanceClass,
  Instance,
  InstanceType,
  InstanceSize,
  SubnetType,
  Peer,
} from '@aws-cdk/aws-ec2'
import { App, Stack, StackProps } from '@aws-cdk/core'

const APP_PORT = 3000

export class AStack extends Stack {
  constructor(app: App, id: string, props?: StackProps) {
    super(app, id, props)

    // Create VPC
    const vpc = new Vpc(this, 'vpc')

    // Creat Security Group for web
    const sgWeb = new SecurityGroup(this, 'sg-web', { vpc })

    // Add Ingress Rule
    sgWeb.addIngressRule(Peer.anyIpv4(), Port.tcp(80))
    sgWeb.addIngressRule(Peer.anyIpv6(), Port.tcp(80))

    // Create SEcurity Group for app
    const sgAppAllowed = new SecurityGroup(this, 'sg-app-allowed', { vpc })
    
    // Create Security Group for app
    const sgApp = new SecurityGroup(this, 'sg-app', { vpc })

    // Add Ingress Rule 
    sgApp.addIngressRule(sgAppAllowed, Port.tcp(APP_PORT))

    // Create EC2 instance for web
    const webServer = new Instance(this, 'web-server', {
      instanceType: InstanceType.of(InstanceClass.T2, InstanceSize.MICRO),
      machineImage: new AmazonLinuxImage({
        generation: AmazonLinuxGeneration.AMAZON_LINUX_2,
      }),
      securityGroup: sgAppAllowed,
      vpc,
      vpcSubnets: {
        subnetType: SubnetType.PUBLIC,
      },
    })

    // Create EC2 instance for app
    const appServer = new Instance(this, 'app-server', {
      instanceType: InstanceType.of(InstanceClass.M5, InstanceSize.LARGE),
      machineImage: new AmazonLinuxImage({
        generation: AmazonLinuxGeneration.AMAZON_LINUX_2,
      }),
      securityGroup: sgApp,
      vpc,
      vpcSubnets: {
        subnetType: SubnetType.PRIVATE,
      },
    })
  }
}
```

```bash
# Dry run
$ yarn cdk synth

# Deploy stack
$ yarn cdk deploy

# Destroy stack
$ yarn deploy
```
