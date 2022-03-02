- [Materials](#materials)
- [CLB vs NLB vs ALB](#clb-vs-nlb-vs-alb)
- [ALB, ALB Listener, ASG, ASG Policy, Launch Configuration, TG Relation Ships](#alb-alb-listener-asg-asg-policy-launch-configuration-tg-relation-ships)
- [connection draining](#connection-draining)

----

# Materials

* [AWS Application Load Balancer](https://jayendrapatil.com/tag/alb/)
* [AWS Load Balancers - How they work and differences between them](http://chinomsoikwuagwu.com/2020/02/14/AWS-Load-Balancers_How-they-work-and-differences-between-them/)
  * ELB Dive Deep 

# CLB vs NLB vs ALB

* [Elastic Load Balancing features](https://aws.amazon.com/elasticloadbalancing/features/?nc1=h_ls)

# ALB, ALB Listener, ASG, ASG Policy, Launch Configuration, TG Relation Ships

* [[AWS CloudFormation] #7 Auto Scaling Group 만들기 ( + Launch Configuration, ALB )](https://honglab.tistory.com/89)

----

ALB 를 제대로 이해하기 위해 ALB, ALB Listener, ASG, ASG Policy, Launch Configuration, TG 의 관계를 파악하는 것이 중요하다. Cloud Formation 으로 이해해 보자.

> Parameter 를 제작한다.

```yaml
Parameters:
  Key:
    Description: KeyPair
    Type: AWS::EC2::KeyPair::KeyName
  
  VPC:
    Description: VPC
    Type: AWS::EC2::VPC::Id

  WebSG:
    Description: SG for Web Server
    Type: AWS::EC2::SecurityGroup::Id
  
  PublicSubnet1:
    Description: public 1
    Type: AWS::EC2::Subnet::Id
  PublicSubnet2:
    Description: public 2
    Type: AWS::EC2::Subnet::Id
  PrivateSubnet1:
    Description: private 1
    Type: AWS::EC2::Subnet::Id
  PrivateSubnet2:
    Description: private 2
    Type: AWS::EC2::Subnet::Id
```

> Launch Configuration 을 제작한다. ASG 는 Launch Configuration 이 필요하다.

```yaml
Resources:
  LC:
    Type: AWS::AutoScaling::LaunchConfiguration
    Properties:
      ImageId: [AMI ID]
      InstanceType: t3.micro
      KeyName: !Ref Key
      LaunchConfigurationName: webserverLC
      SecurityGroups:
        - !Ref WebSG
```

> ALB, TG, ALB Listener 를 제작한다. ALB 는 ALB Listener 를 갖는다. ALB Listener 는 TG 를 갖는다.

```yaml
Resources:
  ALB:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: webserverALB
      Type: application
      SecurityGroups:
        - !Ref WebSG
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2

  TargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      VpcId: !Ref VPC
      Name: webtest
      Port: 80
      Protocol: HTTP
      VpcId: !Ref VPC
      HealthCheckEnabled: true
      HealthCheckIntervalSeconds: 10
      HealthCheckPath: /
      HealthCheckProtocol: HTTP
      HealthCheckTimeoutSeconds: 5
      HealthyThresholdCount: 2
      UnhealthyThresholdCount: 2

  ALBListener:
    Type: AWS::ElasticLoadBalancingV2::Listener
    Properties:
      DefaultActions:
        - Type: forward
          TargetGroupArn: !Ref TargetGroup   
      LoadBalancerArn: !Ref ALB
      Port: 80
      Protocol: HTTP
```

> ASG, ASG Policy 를 제작한다. ASG 는 Launch Configuration, TG 를 갖는다. ASG Policy 는 ASG 를 갖는다.

```yaml
Resources:
  ASG:
    Type: AWS::AutoScaling::AutoScalingGroup
    Properties:
      AutoScalingGroupName: webserverASG
      VPCZoneIdentifier:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
      Cooldown: 10
      LaunchConfigurationName: !Ref LC
      MaxSize: 4
      MinSize: 2
      DesiredCapacity: 2
      TargetGroupARNs:
        - !Ref TargetGroup
      Tags:
        - Key: Name
          Value: web-asg
          PropagateAtLaunch: true

  ASGPolicy:
    Type: AWS::AutoScaling::ScalingPolicy
    Properties:
      AutoScalingGroupName: !Ref ASG
      PolicyType: TargetTrackingScaling
      TargetTrackingConfiguration:
        PredefinedMetricSpecification:
          PredefinedMetricType: ASGAverageCPUUtilization
        TargetValue: 5
```

# connection draining

먼저 Target Group 의 [Deregistration delay @ AWS](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-target-groups.html#deregistration-delay) 와 [Slow start mode @ AWS](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-target-groups.html#slow-start-mode)

[Deregistration delay @ AWS](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-target-groups.html#deregistration-delay) 는 ALB 에서 해제될 Target Group 의 설정이다. Target Group 이 ALB 에서 Deregistration 되는데 걸리는 최대 시간이다. 기본 값은 `300s` 이다. 이 것이 `0` 이면 바로 Deregistration 할 것이다. 이 것이 `0` 보다 크면 그 시간 만큼 기다리고 Deregistration 할 것이다. 

Deregistration 된다는 말은 ALB 에서 더이상 request 를 받지 않는 것을 의미한다. Deregistration 이 시작되면 Target Group 의 instance 들의 상태는 **draining** 으로 바뀐다. 만약 Target Group 의 instance 에서 처리되고 있는 HTTP request 가 있다면 그 것이 처리되기를 Deregistration delay 만큼 기다릴 것이다. Web socket 이라면 Deregistration delay 만큼 기다린 후 connection 을 끊을 것이다???

재밌는 것은 NLB 의 경우 Deregistration delay 이후에 connection 을 끊지 않게 할 수도 있다. [deregistration_delay.connection_termination.enabled @ awscli](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/elbv2/modify-target-group-attributes.html#options)

[Slow start mode @ AWS](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-target-groups.html#slow-start-mode) 는 ALB 에 부착될 Target Group 의 설정이다.
Target Group 이 ALB 에 Registration 되고 나서 full-request 를 받기 시작할 시간이다. 기본 값은 `0s` 이다. 부착되자 마자 바로 HTTP request 를 받겠다는 의미이다. Target Group 의 instance 들에게 warm-up 할 시간을 주고 싶다면 사용한다. 
