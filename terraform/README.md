# Abstract

# Materials

* [Learn Terraform](https://learn.hashicorp.com/terraform)

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
  