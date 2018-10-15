# Abstract

Gamma Color Space 와 Linear Color Space 에 대해 적는다.

# Materials

* [GAMMA AND LINEAR SPACE - WHAT THEY ARE AND HOW THEY DIFFER](http://www.kinematicsoup.com/news/2016/6/15/gamma-and-linear-space-what-they-are-how-they-differ)
* [UNDERSTANDING GAMMA CORRECTION](https://www.cambridgeincolour.com/tutorials/gamma-correction.htm)

# Color Space

![](https://cdn.cambridgeincolour.com/images/tutorials/gamma_chart1e.png)

인간의 눈은 밝은 부분보다는 어두운 부분에 민감하다. 만약 카메라로
사진을 찍어서 파일로 저장할 때 어두운 부분에 밝은 부분보다 많은 비트를
할당하면 사진의 용량을 줄일 수 있다.  이렇게 줄이는 과정을 감마 보정
(gamma correct) 했다고 표현한다. 반대로 이 파일을 모니터로 볼 때
모니터에서 감마 디코딩을 해야 제대로 된 밝기의 사진을 볼 수 있게 된다.

감마인코딩의 방법중 MS가 개발한 sRGB (standard Red Green Blue) 가
산업표준처럼 이용되고 있다. 우리가 보통 포토샵에서 그림을 png 형식으로
저장하게 되면 sRGB 형식으로 데이터가 저장된다.

렌더링 파이프 라인을 감마 색공간으로 하느냐 선형 색공간으로 하느냐에
따라 우리 눈으로 보는 결과물은 달라진다. 물론 선형 색공간이 훨씬 현실
세계에 가까운 결과를 보여 주지만 모든 플래폼에서 동작하는 것은
아니다. 감마 색공간에 비해 연산량이 많아서 저사양의 플래폼에서는
불가능하다.

![](https://static1.squarespace.com/static/525ed7ade4b0924d2f499223/t/575f42e327d4bdc48f2261e4/1465860851928/An+image+comparing+gamma+and+linear+pipelines?format=750w)

위의 그림은 유니티를 예로 색공간을 감마와 선형으로 선택했을 때
렌더링파이프 라인이 어떻게 달라지는지 표현한 것이다.

감마색공간의 경우 gamma correct 된 색정보가 쉐이더에게 입력데이터로
전달되고 연산을 하고 난 후 프레임 버퍼에 기록된다.  결과물의 정확도는
현실세계와 가깝다고 할 수 없다.

선형색공간의 경우 gamma correct 가 제거된 색정보가 쉐이더에게
입력데이터로 전달되고 연사을 하고 난 후 유니티가 다시 gamma correct
하여 프레임 버퍼에 기록하게 된다. 결과물의 정확도는 현실세계와
가깝다고 할 수 있다.
