- [Requirements](#requirements)
  - [Functional requirements](#functional-requirements)
  - [Non-functional requirements](#non-functional-requirements)
- [Capacity Estimation and Constraints](#capacity-estimation-and-constraints)
- [High-level Architecture](#high-level-architecture)
- [Log-level Architecture](#log-level-architecture)
- [System Extension](#system-extension)
- [Q&A](#qa)
- [References](#references)

------

# Requirements

This is system should suuport Availability, Reliability and Durability, Scalability.

## Functional requirements

* Users can upload, download files from any device.
* Users can share files or folders with others.
* This system supports automatic synchronization between devices
* This system supports GB files.
* This system supports ACID.
* This system supports offline editing. As soon as devices got online, all changes will be synced to the remote servers and other devices.
* This system supports revisions of files so users can go back to previous version.

## Non-functional requirements

* This system supports huge read and write.
* Read to write ratio is same.
* Files will be divided by chunks (4MB).
* This system reduce the amount of data exchange by trasnferring updated chunks only.
* This system reduce the storage space and bandwidth by removing duplicated chunks.
* The client keep a local copy of the metdata(file name, size, etc...)
* The client upload the diffs instead of the whole chunk.

# Capacity Estimation and Constraints

| Number                                       | Description      |
| -------------------------------------------- | ---------------- |
| 500 M   | Total Users |
| 200 | Files per user |
| 100 KB   | Average file size |
| 100 billion   | Total files |
| 10 PB   | Total Storage |

# High-level Architecture

# Log-level Architecture

# System Extension

# Q&A

# References
