# The Graphics Hardware Pipeline

- [The Cg Tutorial, Chapter 1: Introduction](http://download.nvidia.com/developer/cg/Cg_Tutorial/Chapter_1.pdf)를 
  읽고 정리한다.

# unity3d rendering pipeline

- [Optimizing graphics rendering in Unity games](https://unity3d.com/kr/learn/tutorials/temas/performance-optimization/optimizing-graphics-rendering-unity-games?playlist=44069)와
  [unity3d rendering pipeline](https://www.youtube.com/watch?v=qHpKfrkpt4c)를 읽고 정리한다.
- CPU는 GPU에게 command들의 덩어리를 전송한다. 이때 전송되는 하나의 단위를 
  batch라고 한다. batch는 draw call, set VB, set IB, 
  Set Transform, Set Shader, Set Texture, Set Blending, 
  Set Z enable 등등을 포함한다.
- draw call은 opengl의 경우 glDrawArrays, glDrawElements 과 같은
  함수 호출과 같다. directx의 경우 Draw* 함수 호출과 같다.
- Set Shader, Set Texture, Set Blending, Set Z enable등을 
  SetPass Call이라고 한다. 즉 SetPass call은 render state을 
  바꾸는 command들이다.
- 10개의 오브젝트가 있다고 해보자. 이때 모두 같은 머터리얼을 사용한다면
  draw call이 10개이고 setpass call은 1개이기 때문에 batch는 11개이다.
  하지만 모두 다른 머터리얼을 사용한다면 draw call이 10개이고 setpass call은
  10개이기 때문에 batch는 20개이다. 
- 만약 10개의 오브젝트를 rendering하기 위해 1개의 batch에 command들을 
  잘 구성할 수 있다면 10개의 batch보다 효율적이다. 이러한 행위를 batching이라고 한다.
- GPU가 CPU로 부터 넘겨받은 batch의 command가 SetPass Call이라면 GPU는 
  renderstate을 갱신하고 Draw Call이라면 설정된 shader에 의해 mesh를
  rendering한다.
- CPU입장에서 GPU에게 command들을 전송할 때 가장 비용이 큰 command는
  SetPass call이다. 따라서 CPU bound인 경우 SetPass call command를
  줄이는 것은 성능향상의 가장 좋은 방법이다.
