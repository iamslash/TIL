- [Abstract](#abstract)
- [Materials](#materials)
- [Basics](#basics)
  - [Simple one](#simple-one)
  - [How to separate develop, production environments](#how-to-separate-develop-production-environments)
  - [Directory Structures](#directory-structures)
  - [Best practices](#best-practices)
  - [How to Debug](#how-to-debug)
  - [How to test with snippet](#how-to-test-with-snippet)
  - [tfstate files](#tfstate-files)
  - [How to use usedata and template](#how-to-use-usedata-and-template)
  - [How to get private ips from ENIs](#how-to-get-private-ips-from-enis)

----

# Abstract

# Materials

* [Learn Terraform](https://learn.hashicorp.com/terraform)
* [AWS Provider Examples @ github](https://github.com/hashicorp/terraform-provider-aws/tree/master/examples)
* [Basic Examples @ github](https://github.com/diodonfrost/terraform-aws-examples)

# Basics

* useful commands

```bash
$ terraform -help
$ terraform -help plan
```

## Simple one

* simple.tf

  ```terraform
  provider "aws" {
    profile    = "default"
    region     = "us-east-1"
  }

  resource "aws_instance" "example" {
    ami           = "ami-b374d5a5"
    instance_type = "t2.micro"
  }
  ```
* run

  ```bash
  $ cd simple
  $ vim simple.tf
  # Initialize a Terraform working directory
  $ terraform init
  $ terraform plan
  $ terraform apply .
  $ terraform destroy .
  ```
  
## How to separate develop, production environments

* [Separate Development and Production Environments](https://learn.hashicorp.com/tutorials/terraform/organize-configuration)

----

There are 2 ways including directories, workspaces. I prefer directories because it's easy to make mistakes for workspaces.

## Directory Structures

```
├── README.md
├── dev
│   ├── aws_instance
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── versions.tf
│   ├── etcd
│   │   ├── main.tf
│   ├── m3dbnode
│   │   ├── main.tf
│   └── network_interface
│       ├── main.tf
├── modules
│   ├── eni
│   │   ├── main.tf
│   │   └── variables.tf
│   ├── etcd
│   │   ├── etcd_ec2.tf
│   │   ├── outputs.tf
│   │   ├── scripts
│   │   │   └── install_etcd.sh.tpl
│   │   └── variables.tf
│   ├── m3dbnode
│   │   ├── m3dbnode_ec2.tf
│   │   ├── outputs.tf
│   │   ├── scripts
│   │   │   └── install_m3dbnode.sh.tpl
│   │   └── variables.tf
│   ├── m3query
│   │   ├── m3query_alb.tf
│   │   ├── m3query_asg.tf
│   │   ├── outputs.tf
│   │   ├── scripts
│   │   │   └── install_m3query.sh.tpl
│   │   └── variables.tf
│   └── prometheus
│       ├── outputs.tf
│       ├── prometheus_ec2.tf
│       ├── scripts
│       │   └── install_prometheus.sh.tpl
│       └── variables.tf
├── prod
│   ├── aws_instance
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── versions.tf
│   ├── etcd
│   │   └── main.tf
│   ├── m3dbnode
│   │   └── main.tf
│   └── network_interface
│       ├── main.tf
└── snippets
    └── get_ips_of_eni
        ├── main.tf
```

## Best practices

* [10 Terraform Best Practices for Better Infrastructure Provisioning](https://geekflare.com/terraform-best-practices/)

## How to Debug

```bash
$ TF_LOG=DEBUG terraform apply
```

## How to test with snippet

You can print out using `output`.

```go
terraform {
  required_version = ">= 0.12"
}

provider "aws" {
  region  = "ap-northeast-2"
  profile = "fastrollback"
}

resource "null_resource" "etcd_initial_cluster_string" {
  count = length(data.aws_network_interface.etcd.*.private_ips)
  triggers = {
    // "etcd10=http://etc10.iamslash.com:2380"
    value = format("etcd%d=https://%s:2380", count.index + 1, sort(element(data.aws_network_interface.etcd.*.private_ips, count.index))[0])
  }
}

data "aws_network_interfaces" "etcd" {
  count    = var.etcd_instance_count

  tags = {
    Name  = replace(format("%s-%s%03d-eni", var.tag_role, var.etcd_app_name, var.etcd_instance_seed_no + count.index), "_", "-")
    role  = var.tag_role
    owner = var.tag_owner
    env   = var.tag_env
  }
}

data "aws_network_interface" "etcd" {
  count   = var.etcd_instance_count

  id = sort(element(data.aws_network_interfaces.etcd.*.ids, count.index))[0]
}

output "instance_ips" {
  value = trimsuffix(join(",", null_resource.etcd_initial_cluster_string.*.triggers.value), ",")
}
```

## tfstate files

*.tfstate fils include resources after apply. We can manage this file with uploading to AWS S3. Besides when we use AWS DynamoDB, we can manage lock of provision. But that's too complicated. 

I will provision for first provision and manage by manual. I think using seed_no, count is a good solution. For example think about using 2 arguments including `etcd_instance_seed_no`, `etcd_instance_count`. You can make instances easily. But this is a one way management because it's difficult destroy resources.

## How to use usedata and template

## How to get private ips from ENIs


