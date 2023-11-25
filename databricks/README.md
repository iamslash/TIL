- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [Features](#features)
  - [Databricks Lakehouse](#databricks-lakehouse)

-----

# Abstract

**Databricks** is a company that provides a cloud-based platform for data
engineering, data science, machine learning, and analytics. Their platform, also
called Databricks, is built on top of Apache Spark, an open-source distributed
computing framework designed to process large-scale data. The platform offers a
unified workspace for data processing, collaborative notebook environment, and
tools for visualizing and managing data pipelines. Databricks aims to simplify
big data processing and help organizations derive insights from their data more
quickly and efficiently.

# Materials

- [DataBricks Ebooks](https://www.databricks.com/resources?_sft_resource_type=ebook)
  - Ebooks for data engineerings including databricks solutions.

# Basic

## Features

Databricks는 빅데이터 처리와 분석, 머신러닝, 데이터 과학을 위한 통합
플랫폼입니다. 이 플랫폼은 주로 아래와 같은 기능을 제공합니다:

- 데이터 처리 및 ETL (Extract, Transform, Load): Databricks는 대량의 데이터를
  처리하고 정제하는 강력한 기능을 제공합니다. [Apache Spark](/spark/README.md)를
  기반으로 한 이 플랫폼은 데이터를 추출하고, 변환하며, 다양한 데이터 소스로부터
  데이터를 적재하는 과정을 자동화하고 최적화합니다.
- 실시간 데이터 분석: 실시간으로 대규모 데이터 세트를 분석하는 데 적합합니다.
  사용자는 빠른 시간 내에 인사이트를 얻을 수 있으며, 대화형 쿼리를 실행할 수
  있습니다.
- 머신러닝 및 AI: Databricks는 머신러닝과 인공지능 개발을 지원합니다. MLflow라는
  머신러닝 라이프사이클 관리 도구를 통해 모델 훈련, 실험 추적, 배포 및
  모니터링을 손쉽게 할 수 있습니다.
- 데이터 과학과 협업: 데이터 과학자, 분석가, 엔지니어 간의 협업을 촉진합니다.
  주피터 노트북 스타일의 인터페이스를 제공하여 코드, 데이터, 인사이트를 팀원들과
  공유하고 협업할 수 있습니다.
- 다양한 데이터 소스와의 통합: Databricks는 다양한 데이터 소스와 쉽게
  통합됩니다. 이를 통해 기업들은 여러 데이터 저장소에서 데이터를 수집하고 통합된
  뷰를 생성할 수 있습니다.
- 클라우드 기반의 유연성: AWS, Azure, Google Cloud Platform과 같은 주요 클라우드
  서비스 제공업체에서 호스팅됩니다. 이는 사용자에게 높은 확장성과 유연성을
  제공합니다.
- 보안 및 관리: 기업 수준의 보안 기능을 제공하며, 사용자 및 데이터 액세스 관리를
  위한 포괄적인 도구를 갖추고 있습니다.

Databricks는 이러한 기능을 통해 데이터 중심의 인사이트를 생성하고, 비즈니스
의사결정 과정을 지원하며, 데이터 기반의 솔루션 개발을 가능하게 합니다.

## Databricks Lakehouse

**Databricks Lakehouse** is a specific solution offered by Databricks. It refers
to their architecture that combines the best features of **data lakes** and
**data warehouses** to create a unified platform. This platform caters to
diverse data needs, including data engineering, data science, machine learning,
and analytics.

**Databricks Lakehouse** is built on several key technologies, such as:

- **Apache Spark**: A fast and general-purpose cluster-computing system for big
  data processing and machine learning.
- **Delta Lake**: An open-source storage layer that brings ACID transactions,
  scalability, and reliability to data lakes.
- **MLflow**: An open-source platform for machine learning lifecycle management.

These technologies, combined with Databricks' cloud-based platform, enable
organizations to store, process, and analyze large volumes of data, both
structured and unstructured, in a single, cost-effective, and high-performance
unified solution. The Lakehouse paradigm aims to simplify data management and
accelerate the generation of valuable insights from data.
