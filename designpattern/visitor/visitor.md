# Abstract

Defines a new operation to a class without change.

기능 (Visitor) 이 추상 (Element) 과 분리되어 있다. 따라서 기존의 추상 (Element) 를 변경하지 않고 새로운 기능 (Visitor) 을 추가할 수 있다. 

그러나 새로운 추상 (Element) 가 추가된다면 기존의 코드를 수정할 수 밖에 없다.

Concrete Visitor Class 는 `Accept(Visitor v)` Method 를 갖는다. Concret Element Class 는 `Visit(Element e)` Method 를 갖는다.

# Materials

* [Visitor](https://www.dofactory.com/net/visitor-design-pattern)
* [방문자 패턴 - Visitor pattern](https://thecodinglog.github.io/design/2019/10/29/visitor-pattern.html)
* [Visitor Design Pattern in Java @ baeldung](https://www.baeldung.com/java-visitor-pattern)
* [Double Dispatch in DDD @ baeldung](https://www.baeldung.com/ddd-double-dispatch)
  
# Concept Class Diagram

> [src](visitor.puml)

![](visitor.png)

# Idea

고객의 등급별로 혜택을 주는 것을 구현해 보자.

```java
public interface Member {
}
public class Gold implements Member {
  public void point() {
    System.out.println("Gold::point");    
  }
  public void discount() {
    System.out.println("Gold::discount");    
  }
}
public class Silver implements Member {
  public void point() {
    System.out.println("Silver::point");    
  }
  public void discount() {
    System.out.println("Silver::discount");    
  }
}
public class Main {
  public static void main(String[] args) {
    Member gold = new Gold();
    Member silver = new Silver();

    gold.point();
    silver.point();
    gold.discount();
    silver.discount();
  }
}
```

다음과 같은 문제점들이 있다.

* Member 들에게 point, discount 와 같은 혜택을 주고 싶을 때 iterator 를 이용하여 순회할 수 없다???
* point, discount 외에 coupon 이란 혜택을 추가하고 싶다. 모든 Concrete Member class 가 새로운 혜택을 구현했다는 보장이 없다???

앞서 언급한 문제들을 해결하기 위해 Benefit class 를 만든다.

```java
public interface Benefit {
    void point();
    void discount();
}
public interface Member extends Benefit{ }
```

이것은 다음과 같은 문제점들이 있다.

* point, discount 외에 coupon 이란 헤택을 추가하고 싶다. Concret Member Class 를 변경해야 한다. SRP (Single Respoinsibility Principal) 에 어긋난다.

Member 와 Benefix 을 분리해 보자.

```java
public interface Member {}
public class Gold implements Member {}
public class Silver implements Member {}
public interface Benefit {
  void point(Member member);
  void discount(Member member);
}
public class BenefitImpl implements Benefit {
  @Override
  public void point(Member member) {
    if (member instanceof Gold) {
      System.out.println("BenefitImpl::point, Gold");
    } else if (member instanceof Silver) {
      System.out.println("BenefitImpl::point, Silver");
    }
  }
  @Override
  public void discount(Member member) {
    if (member instanceof Gold) {
      System.out.println("BenefitImpl::discount, Gold");
    } else if (member instanceof Silver) {
      System.out.println("BenefitImpl::discount, Silver");
    }
  }
}
public class Main {
  public static void main(String[] args) {
    Benefit benefit = new BenefitImpl();
    Member gold = new Gold();
    Member silver = new Silver();

    benefit.point(gold);
    benefit.point(silver);
    benefit.discount(gold);
    benefit.discount(silver);
  }
}
```
 
이것은 다음과 같은 문제가 있다.

* Green Member Class 가 추가되었을 때 다음과 같이 instanceof 구문이 늘어난다.

  ```java
  public class BenefitImpl implements Benefit {
    @Override
    public void point(Member member) {
      if (member instanceof Gold) {
        System.out.println("BenefitImpl::point, Gold");
      } else if (member instanceof Silver) {
        System.out.println("BenefitImpl::point, Silver");
      } else if (member instanceof Green) {
        System.out.println("BenefitImpl::point, Green");
      }
    }
    @Override
    public void discount(Member member) {
      if (member instanceof Gold) {
        System.out.println("BenefitImpl::discount, Gold");
      } else if (member instanceof Silver) {
        System.out.println("BenefitImpl::discount, Silver");
      } else if (member instanceof Green) {
        System.out.println("BenefitImpl::discount, Green");
      }
    }
  }
  ```
  
* coupon 이라는 혜택을 추가한다면 다음과 같이 instanceof 구문이 늘어난다.

  ```java
  public class BenefitImpl implements Benefit {
    @Override
    public void point(Member member) {
      if (member instanceof Gold) {
        System.out.println("BenefitImpl::point, Gold");
      } else if (member instanceof Silver) {
        System.out.println("BenefitImpl::point, Silver");
      }
    }
    @Override
    public void discount(Member member) {
      if (member instanceof Gold) {
        System.out.println("BenefitImpl::discount, Gold");
      } else if (member instanceof Silver) {
        System.out.println("BenefitImpl::discount, Silver");
      }
    }
    @Override
    public void coupon(Member member) {
      if (member instanceof Gold) {
        System.out.println("BenefitImpl::coupon, Gold");
      } else if (member instanceof Silver) {
        System.out.println("BenefitImpl::coupon, Silver");
      }
    }  
  }
  ```

이것을 해결하기 위해 Benefit 에 Concret Member Class 를 argument 로 받도록 수정해보자.

```java
public interface Benefit {
  void point(Gold member);
  void point(Silver member);
  void discount(Gold member);
  void discount(Silver member);
}
public class BenefitImpl implements Benefit {
  @Override
  public void point(Gold member) {
    System.out.println("BenefitImpl::point, Gold");
  }
  @Override
  public void point(Silver member) {
    System.out.println("BenefitImpl::point, Silver");
  }
  @Override
  public void discount(Gold member) {
    System.out.println("BenefitImpl::discount, Gold");
  }
  @Override
  public void discount(Silver member) {
    System.out.println("BenefitImpl::discount, Silver");
  }
}
public class Main {
  public static void main(String[] args) {
    Benefit benefit = new BenefitImpl();
    Member gold = new Gold();
    Member silver = new Silver();

    benefit.point(gold);
    benefit.point(silver);
    benefit.discount(gold);
    benefit.discount(silver);
  }
}
```

이것은 다음과 같은 문제점들이 있다.

* `benefit.point(), benefit.discount()` 는 compile error 가 발생한다. `gold, silver` 는 Member type 이다. 그러나 `benefit.point(), benefit.discount()` 는 Member instance 를 전달 받을 수 없다. instanceof 를 해결하기 위해 Benefit 의 `point, discount` 의 prototype 을 변경했기 때문이다.

자 이제 Visitor passtern 을 이용하여 해결해 보자. **Feature** 를 담당하는 Benefit 은 **Visitor** 이고 **Object Structure** 를 담당하는 Member 는 **Element** 이다.

```java
public interface Benefit {
  void getBenefit(Gold member);
  void getBenefit(Silver silver);
}
public class PointBenefit implements Benefit {
  @Override
  public void getBenefit(Gold member) {
    System.out.println("PointBenefit::getBenefit, Gold");
  }
  @Override
  public void getBenefit(Silver member) {
    System.out.println("PointBenefit::getBenefit, Silver");
  }
}
public class DiscountBenefit implements Benefit {
  @Override
  public void getBenefit(Gold member) {
    System.out.println("DiscountBenefit::getBenefit, Gold");
  }
  @Override
  public void getBenefit(Silver member) {
    System.out.println("DiscountBenefit::getBenefit, Silver");
  }
}
public interface Member {
  void getBenefit(Benefit benefit);
}
public class Gold implements Member {
  @Override
  public void getBenefit(Benefit benefit) {
    benefit.getBenefit(this);
  }
}
public class Silver implements Member {
  @Override
  public void getBenefit(Benefit benefit) {
    benefit.getBenefit(this);
  }
}
public class Main {
  public static void main(String[] args) {
    Member gold = new Gold();
    Member silver = new Silver();
    Benefit pointBenefit = new PointBenefit();
    Benefit discountBenefit = new DiscountBenefit();

    gold.getBenefit(pointBenefit);
    silver.getBenefit(pointBenefit);
    gold.getBenefit(discountBenefit);
    silver.getBenefit(discountBenefit);
  }
}
```

이제 새로운 혜택 즉 새로운 기능 coupon 을 추가해 보자.

```java
public class CouponBenefit implements Benefit {
  @Override
  public void getBenefit(Gold member) {
    System.out.println("CouponBenefit::getBenefit, Gold");
  }
  @Override
  public void getBenefit(Silver member) {
    System.out.println("CouponBenefit::getBenefit, Silver");
  }
}
public class Main {
  public static void main(String[] args) {
    Member gold = new Gold();
    Member silver = new Silver();
    Benefit pointBenefit = new PointBenefit();
    Benefit discountBenefit = new DiscountBenefit();
    Benefit couponBenefit = new CouponBenefit();

    gold.getBenefit(pointBenefit);
    silver.getBenefit(pointBenefit);
    gold.getBenefit(discountBenefit);
    silver.getBenefit(discountBenefit);
    gold.getBenefit(couponBenefit);
    silver.getBenefit(couponBenefit);
  }
}
```

Visitor Pattern 을 이용했기 때문에 다음과 같은 특징들을 갖는다.

* 새로운 기능 즉 혜택 coupon 을 추가하기 위해 CouponBenefit 만 추가했다. 새로운 기능 추가를 위해 새로운 Class 만 추가했다.
* Concret Element 가 추가된다면 Benefit 또한 변경되야 한다. 예를 들어 Green Clas 가 추가된다면 Benefit 및 Concret Benefit 이 모두 변경되야 한다. 대상 객체 즉 Element 가 잘 바뀌지 않고 알고리즘 즉, Visitor 가 추가될 가능성이 있을 때 사용한다.
* Visitor Pattern 은 Double Dispatch 를 이용한 것이다. Double Dispatch 는 runtime 에 receiver, parameter type 과 같이 두가지를 고려해서 실행될 method 를 결정하는 것이다.
* [Visitor @ dofactory](https://www.dofactory.com/net/visitor-design-pattern) 의 Real World Example 은 조금 특이하다. ConcretVisitor 의 Visit Method 는 Element Object 를 argument 로 한다. Concrete Element Object 를 argument 로 하지 않는다. Concrete Element Object 별로 다르게 business logic 을 다뤄야 하는 하는 경우를 처리할 수는 없다. Strategy Pattern 과 차이가 없는 것 같다.

# Examples

* [Visitor by go](/golang/designpattern/visitor.md)


