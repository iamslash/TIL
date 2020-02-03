# Materials

* [AWS CloudFormation](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/Welcome.html)
* [AWS CloudFormation CLI](https://docs.aws.amazon.com/cli/latest/reference/cloudformation/index.html)
* [AWS CDK](https://github.com/aws/aws-cdk)
  * AWS Cloud Development Kit is a frameowrk for defining cloud infrastructure in code 
  * [examples]([bash-my-aws https://github.com/bash-my-aws/bash-my-aws](https://github.com/aws-samples/aws-cdk-examples))
* [aws-cloudformation / Awesome CloudFormation @ github](https://github.com/aws-cloudformation/awesome-cloudformation)
  * [JeffRMoore / awesome-cloudformation](https://github.com/JeffRMoore/awesome-cloudformation)

# Examples

* [AWS Quick Start](https://github.com/aws-quickstart)
  * CloudFomration examples including GitHub Enterprise, Consul, MySQL, etc...
  * [quickstart-github-enterprise](https://github.com/aws-quickstart/quickstart-github-enterprise)
  * [quickstart-hashicorp-consul](https://github.com/aws-quickstart/quickstart-hashicorp-consul)
  * [quickstart-amazon-aurora-mysql](https://github.com/aws-quickstart/quickstart-amazon-aurora-mysql)
* [Free Templates for AWS Cloudformation](https://templates.cloudonaut.io/en/stable/)
  * Have [jenkins](https://templates.cloudonaut.io/en/stable/jenkins/) CloudFormation templates
* [GitHub on AWS](https://enterprise.github.com/trial/boot?download_token=5750bba5e67c99905804)
* [Amazon ECS, AWS CloudFormation, and an Application Load Balancer for Jenkins](https://github.com/Kong/jenkins-infrastructure)
* [Challenges Your AWS Cloudformation Skills](https://github.com/dennyzhang/challenges-cloudformation-jenkins)

# Basic Usages

# Tips

## Disable rollback

* CloudFormation Stack 제작을 실패했을 때 기본적으로 RollBack 해버린다. 디버깅을 위해서 다음과 같은 옵션으로 RollBack 을 꺼둘 수 있다.

  ```bash
  $   aws cloudformation create-stack \
  --disable-rollback \
  --region "${AWS_REGION}" \
  --stack-name "${CF_STACK_NAME}" \
  --template-body "${TEMPLATE_BODY}" \
  --capabilities "${CAPABILITIES}" \
  --parameters \
  ...
  ```

## Cloud-init output log

* EC2 instance 를 생성한다면 user-data 를 디버깅할 필요가 있다. 다음과 같은 로그를 참고하자.

  ```bash
  $ vim /var/log/cloud-init.log
  $ vim /var/log/cloud-init-output.log  
  ```