# Basic

```bash
conda --help

conda info                        // 아나콘다 환경 확인

conda create -n cuda pip python=3.6 // 아나콘다 환경 cuda 만들기

conda env list

conda activate cuda                       // 아나콘다 환경 cuda 활성화

conda deactivate cuda                     // 아나콘다 환경 cuda 비활성화

conda remove --name cuda --all      // 아나콘다 환경 cuda 삭제

# Rename env tf to tf2
conda create --name tf2 --clone tf
conda remove --name tf --all
```
