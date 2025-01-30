# Materials

- [GPU는 어떻게 작동할까 | youtube](https://www.youtube.com/watch?v=ZdITviTD3VM)
- [생성형 AI 개발을 위한 NVIDIA GPU 아키텍처의 이해](https://www.samsungsds.com/kr/insights/nvidia-gpu-architecture.html)
- [15.1. Heterogeneous Computing: Hardware Accelerators, GPGPU Computing, and CUDA | DiveIntoSystems](https://diveintosystems.org/book/C15-Parallel/gpu.html)

# Basic

다음은 Simplied GPU Architecture 이다.

![](/csa/img/2024-01-07-18-09-38.png)

GPU 아키텍처는 컴퓨터 그래픽 및 이미지 처리를 위해 설계된 하드웨어로, 주로 비디오 게임 산업에 의해 발전되었습니다. GPU는 여러 개의 SM(Streaming Multiprocessor)로 구성되어 있는데, 각 SM은 32개의 SP(Streaming Processor)로 구성되어 있습니다. 하나의 SP는 코어라고도 부릅니다. 이러한 구조를 통해 병렬 처리를 효율적으로 수행할 수 있습니다.

GPU의 하드웨어 실행 모델은 SIMD의 변형인 SIMT(Single Instruction/Multiple Thread)를 구현하는데, 이는 여러 스레드를 처리하는 데 동시에 실행되는 하나의 명령어를 사용하는 다중 스레딩 SIMD와 같다. WARP는 CUDA에서 32 개의 스레드를 묶은 그룹을 의미하며, 하나의 WARP는 하나의 SM에서 실행됩니다. 이러한 WARP를 실행하기 위해 각 SM에는 WARP 스케줄러(Warp Scheduler)가 존재합니다.

WARP 스케줄러는 SM에 포함되어 있으며, 실행해야 할 WARP를 선택하고 각 SP 코어에 할당하는 역할을 합니다. 이를 통해 동시에 여러 WARP가 실행되어 병렬성이 향상되고, 자원이 효율적으로 사용됩니다.

락스텝(Lockstep) 실행을 사용하여 각 스레드는 동일한 주기에 동일한 명령어를 실행하게 되지만 다른 데이터에서 작업하게 됩니다. 이를 통해 병렬로 다수의 픽셀 업데이트가 가능해집니다. 이러한 실행 방식 덕분에 프로세서의 설계가 단순화되어 동일한 명령어 제어 유닛을 공유할 수 있습니다.

SM은 공유 메모리, 레지스터, L1 캐시 등 자체 실행 제어 유닛과 메모리 공간을 갖추고 있어, 각 코어가 효과적으로 동작할 수 있도록 지원합니다. 이런 구조와 실행 방식, WARP 스케줄러 덕분에 GPU 아키텍처는 병렬 처리를 통해 다양한 그래픽 및 이미지 작업에 높은 성능을 발휘할 수 있습니다.
