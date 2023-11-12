- [Abstract](#abstract)
- [Materials](#materials)

-----

# Abstract

**MLflow** is an open-source platform for managing the end-to-end machine learning
lifecycle. It was developed by Databricks to help data scientists and developers
easily track experiments, package code into reusable projects, and share and
deploy models across different platforms and environments. MLflow aims to
simplify and standardize the complex process of developing, training,
evaluating, and deploying machine learning models.

MLflow consists of the following main components:

- **MLflow Tracking**: This component helps you keep track of your experiments,
  parameters, metrics, models, and artifacts in a centralized fashion. It
  provides a simple API to log parameters, metrics, and artifacts and query them
  later. It also offers a user interface to visualize and compare experiment
  results.
- **MLflow Projects**: This component helps you package and share code in a
  reusable and reproducible format. MLflow projects can be linked to a GitHub
  repository or stored in a local directory. It defines the required
  dependencies, entry points, and execution environments, making it easy for
  others to reuse the code and run the same experiment with the same setup in a
  consistent manner.
- **MLflow Models**: This component standardizes the process of packaging and
  deploying machine learning models, allowing you to easily switch between
  different models and frameworks. With MLflow models, you can save models in a
  variety of formats and later serve them using REST APIs, containerization
  (e.g., Docker), or cloud-based deployment options such as AWS SageMaker or
  Azure ML.
- **MLflow Model Registry**: This component enables you to organize, track, and
  version different models, making it easy to manage and collaborate on models
  throughout their lifecycle. It provides a centralized interface to register,
  store, and manage a wide variety of model formats across multiple development
  stages, from experimentation to production.

By employing MLflow, organizations can streamline their machine learning
workflows, enforce best practices, and improve collaboration between data
scientists and engineers, ultimately speeding up the process of delivering
data-driven insights and solutions.

# Materials

- [mlflow | github](https://github.com/mlflow/mlflow/)

