
# intro

- unity3d 최적화에 대해 경험해 본 것들 위주로 적어본다. 

# usage

- [Optimizing graphics rendering in Unity games](https://unity3d.com/kr/learn/tutorials/temas/performance-optimization/optimizing-graphics-rendering-unity-games?playlist=44069)
  - rendering pipeline, profile window, frame debugger, 
    multithreaded rendering, graphics jobs, draw call, setpass call
    , batch, batching, frustum culling, layer cull distances, occlusion culling,
    shadow distance, reflection probe, static batching, dynamic batching,
    gpu instancing, texture atlasing, combined mesh, shared material,
    skinned mesh renderer, bake mesh, fill rate, overdraw,
    texture compression, mipmap, normap mapping, level of detail 등등의
    내용을 자세히 기술한다.
- player settings옵션을 꼼꼼하게 확인해 본다.
  - release빌드할때 script optimization등등을 신경쓴다.
- Resources 폴더의 내용은 runtime memory를 잡아 먹는다.
- Assets폴더에 존재하는 모든 script들은 build package에
  포함된다. 사용하지 않는 script들은 삭제하자.
  - [참고](https://unity3d.com/kr/learn/tutorials/temas/best-practices/resources-folder)
- json을 처리할때 minijson보다 JsonUtility를 사용하자. runtime memory를 절약할 수 있다.
- Runtime에 fps 를 줄이면 battery life 를 증가시킬 수 있다.
- Buildtime에 DPI 를 줄이면 battery life 를 증가시킬 수 있다.
- unity 는 기본적으로 하나의 main thread, 하나의 render thread, 다수의 worker thread 가 동작한다. 이때 render thread 는 GPU command 를 GPU 에게 보내는       일을 하는데 만약 render thread 에서 bottle neck 현상이 발생한다면 SetPass       call 의 개수를 줄이는 최적화를 해야 한다. GPU command 중 SetPass call 이 가장 비용이 크기 때문이다.
- main thread 는 rednering 의 중요한 단계들을 처리한다. 따라서 main thread 가 수행하는 일들 중 rendering 과 관련없는 것을 work thread 가 처리하게 하면 main thread 의 성능을 개선할 수 있다.
