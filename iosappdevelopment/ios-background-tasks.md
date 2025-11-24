- [Overview](#overview)
- [iOS Background Task 처리 방법 종합 가이드](#ios-background-task-처리-방법-종합-가이드)
  - [iOS Background 실행 방법 전체 목록](#ios-background-실행-방법-전체-목록)
    - [1. BGTaskScheduler Framework (iOS 13+)](#1-bgtaskscheduler-framework-ios-13)
    - [2. Background Modes (Continuous Background)](#2-background-modes-continuous-background)
    - [3. Background Transfer](#3-background-transfer)
    - [4. Push Notifications](#4-push-notifications)
    - [5. Background Task Completion (Legacy, iOS 4+)](#5-background-task-completion-legacy-ios-4)
    - [6. 특수 Framework Background](#6-특수-framework-background)
    - [실제로 자주 사용되는 방법](#실제로-자주-사용되는-방법)
    - [사진 Moderation \& Upload 앱 권장 조합](#사진-moderation--upload-앱-권장-조합)
  - [Background Task 유형별 상세 설명](#background-task-유형별-상세-설명)
    - [1. Background App Refresh (BGAppRefreshTask)](#1-background-app-refresh-bgapprefreshtask)
    - [2. Background Processing Task (BGProcessingTask)](#2-background-processing-task-bgprocessingtask)
    - [3. URLSession Background Transfer](#3-urlsession-background-transfer)
    - [4. Background Task Completion (Legacy)](#4-background-task-completion-legacy)
    - [5. Silent Push Notifications](#5-silent-push-notifications)
  - [AppDelegate 통합 설정](#appdelegate-통합-설정)
  - [Info.plist 설정](#infoplist-설정)
- [실전 예제: 대량 사진 Moderation \& Upload](#실전-예제-대량-사진-moderation--upload)
  - [1. CoreML Moderation Model Service](#1-coreml-moderation-model-service)
  - [2. Photo Processing Service](#2-photo-processing-service)
  - [3. Upload Service (Background URLSession)](#3-upload-service-background-urlsession)
  - [4. Processing Cache (진행 상태 저장)](#4-processing-cache-진행-상태-저장)
  - [5. Background Task Integration](#5-background-task-integration)
  - [6. UI Integration (SwiftUI)](#6-ui-integration-swiftui)
- [iOS 18 vs iOS 26 GPU 할당](#ios-18-vs-ios-26-gpu-할당)
  - [iOS 18: CPU 중심 처리](#ios-18-cpu-중심-처리)
    - [제약사항](#제약사항)
    - [iOS 18 최적화 전략](#ios-18-최적화-전략)
    - [iOS 18 성능 최적화](#ios-18-성능-최적화)
  - [iOS 26: GPU 백그라운드 지원](#ios-26-gpu-백그라운드-지원)
    - [새로운 기능](#새로운-기능)
    - [iOS 26 구현](#ios-26-구현)
    - [iOS 26 최적화된 Photo Processing](#ios-26-최적화된-photo-processing)
  - [버전별 분기 처리](#버전별-분기-처리)
  - [System Monitor (배터리 \& 열 관리)](#system-monitor-배터리--열-관리)
  - [iOS 26 BGContinuedProcessingTask 구현](#ios-26-bgcontinuedprocessingtask-구현)
    - [BGContinuedProcessingTask 특징](#bgcontinuedprocessingtask-특징)
    - [1. Info.plist 및 Entitlements 설정](#1-infoplist-및-entitlements-설정)
    - [2. Live Activity 정의](#2-live-activity-정의)
    - [3. Live Activity Widget](#3-live-activity-widget)
    - [4. BGContinuedProcessingTask Service](#4-bgcontinuedprocessingtask-service)
    - [5. SwiftUI Integration for iOS 26](#5-swiftui-integration-for-ios-26)
    - [6. AppDelegate 설정](#6-appdelegate-설정)
    - [사용 방법](#사용-방법)
  - [성능 비교표](#성능-비교표)
  - [최종 권장사항](#최종-권장사항)
    - [iOS 18 (BGProcessingTask)](#ios-18-bgprocessingtask)
    - [iOS 26 (BGProcessingTask)](#ios-26-bgprocessingtask)
    - [iOS 26 (BGContinuedProcessingTask) ⭐ 추천](#ios-26-bgcontinuedprocessingtask--추천)
    - [앱별 추천 방식](#앱별-추천-방식)


-----

* [Background Tasks | apple](https://developer.apple.com/documentation/backgroundtasks)
* [Advances in App Background Execution | wwdc2019](https://developer.apple.com/videos/play/wwdc2019/707/)
  * [src](https://developer.apple.com/documentation/backgroundtasks/refreshing_and_maintaining_your_app_using_background_tasks)
* [Background execution demystified | wwdc2020](https://developer.apple.com/videos/play/wwdc2020/10063)
* [[iOS] BackgroundTasks Framework 간단 정리](https://lemon-dev.tistory.com/entry/iOS-BackgroundTask-Framework-%EA%B0%84%EB%8B%A8-%EC%A0%95%EB%A6%AC)
* [How to manage background tasks with the Task Scheduler in iOS 13?](https://snow.dog/blog/how-to-manage-background-tasks-with-the-task-scheduler-in-ios-13)

----

# Overview

iOS 는 `Background Task Completion` 을 제공한다. iOS 13 이전에도 있었던 것 같다.
foreground 의 app 이 background 로 바뀌면 하던 일을 마무리할 수 있다. foreground
에서 background 로 바뀔 때 background 에서 한번 실행된다.

iOS 13 부터 `BGAppRefreshTask`, `BGProcessingTask` 를 제공한다. 

`BGAppRefreshTask` - 비교적 가벼운 logic 이 적당하다. app 이 다음 번에
foreground 가 되었을 때 UI 를 미리 업데이트하는 logic 에 적당하다. 예를 들어
user 가 획득한 점수를 원격으로부터 받아오는 것이 해당된다.

`BGProcessingTask` - 비교적 무거운 logic 이 적당하다. 예를 들어 아주 긴 파일을
다운로드하는 것이 해당된다. 

두 가지 방식에 대해 cancel 조건이 다를 것이다. iOS 가 언제 background task 를
취소할지 예측할 수 없다. 언제 실행될지도 예측할 수 없다. UX 를 신경써야 한다.

테스트 방법은 [Starting and Terminating Tasks During Development |
apple](https://developer.apple.com/documentation/backgroundtasks/starting_and_terminating_tasks_during_development)
을 참고한다. 

`BGTaskScheduler.shared.submit()` 에 break point 를 설정한다. app 의 실행이 멈출
때 LLDB prompt 에 다음과 같은 command line 을 입력하여 background task 를 시작
혹은 종료할 수 있다. test 를 위해 AppStore 제출과 관계없는 code 를 작성할 필요가 있다.

```
LLDB> e -l objc -- (void)[[BGTaskScheduler sharedScheduler] _simulateLaunchForTaskWithIdentifier:@"TASK_IDENTIFIER"]

LLDB> e -l objc -- (void)[[BGTaskScheduler sharedScheduler] _simulateExpirationForTaskWithIdentifier:@"TASK_IDENTIFIER"]
```

# iOS Background Task 처리 방법 종합 가이드

## iOS Background 실행 방법 전체 목록

iOS에서 제공하는 모든 백그라운드 실행 방법을 정리합니다.

### 1. BGTaskScheduler Framework (iOS 13+)
주기적 또는 시스템 최적화 시점에 실행되는 작업

- **BGAppRefreshTask** (iOS 13+) - 30초, 가벼운 작업 (데이터 동기화)
- **BGProcessingTask** (iOS 13+) - 수 분, 무거운 작업 (ML, 대량 데이터 처리)
- **BGContinuedProcessingTask** (iOS 26+) - Foreground에서 시작하여 Background에서 계속 실행
  - Live Activity로 진행 상황 표시
  - 사용자가 취소 가능
  - GPU 접근 가능 (Background GPU Access entitlement 필요)
  - ProgressReporting 프로토콜로 진행률 보고 필수

### 2. Background Modes (Continuous Background)
앱이 백그라운드에서 **지속적으로** 실행되어야 하는 특수 목적

- **Audio** - 백그라운드 오디오 재생 (음악 앱, 팟캐스트)
- **Location Updates** - 위치 추적 (지도, 피트니스 앱)
  - Significant Location Changes - 배터리 효율적
  - Region Monitoring - 지오펜스
  - Visits Monitoring - 사용자 방문 감지
- **VoIP** - VoIP 앱 (FaceTime, Zoom)
- **External Accessory** - 하드웨어 액세서리 통신
- **Bluetooth** - BLE central/peripheral 모드
- **Background Fetch** (Deprecated) - BGAppRefreshTask 사용 권장

### 3. Background Transfer
파일 전송을 백그라운드에서 계속 진행

- **URLSession Background Transfer** - 앱 종료되어도 업로드/다운로드 계속

### 4. Push Notifications
서버에서 앱을 깨워서 작업 실행

- **Silent Push** - `content-available: 1`, 30초
- **Regular Push** - 사용자 알림과 함께

### 5. Background Task Completion (Legacy, iOS 4+)
Foreground → Background 전환 시 마무리 작업

- **beginBackgroundTask** / **endBackgroundTask** - 30초

### 6. 특수 Framework Background
특정 기능을 위한 백그라운드 실행

- **HealthKit Background Delivery** - 건강 데이터 변경 시 앱 깨우기
- **CallKit** - 통화 관련 이벤트 처리
- **HomeKit Automation** - 스마트 홈 자동화 실행
- **WatchKit** - Apple Watch 앱과의 통신
- **PushKit** (Deprecated) - VoIP push 전용 (CallKit으로 대체 권장)

### 실제로 자주 사용되는 방법

**일반 앱:**
1. **BGAppRefreshTask** - 주기적 데이터 동기화
2. **BGProcessingTask** - 무거운 작업 (ML, 대량 데이터)
3. **URLSession Background** - 파일 전송
4. **Silent Push** - 서버 트리거 작업

**특수 목적 앱:**
- **Audio** - 음악/팟캐스트 앱
- **Location** - 지도/피트니스 앱
- **VoIP** - 통화 앱
- **HealthKit** - 건강/피트니스 앱

### 사진 Moderation & Upload 앱 권장 조합

5,000장 사진 처리 앱의 경우:

1. **BGProcessingTask** ⭐ - CoreML 실행 (메인 처리)
2. **URLSession Background Transfer** ⭐ - 사진 업로드
3. **Optional: Silent Push** - 서버에서 처리 시작 트리거

## Background Task 유형별 상세 설명

### 1. Background App Refresh (BGAppRefreshTask)

**특징:**
- 실행 시간: 약 30초
- 용도: 가벼운 작업 (데이터 동기화, UI 업데이트 준비)
- 실행 조건: 시스템이 최적의 시간 선택
- 빈도: 하루 여러 번 가능

**구현 예제:**

```swift
import BackgroundTasks

class BackgroundTaskManager {
    static let shared = BackgroundTaskManager()
    static let refreshTaskID = "com.yourapp.refresh"

    func registerBackgroundTasks() {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: Self.refreshTaskID,
            using: nil
        ) { task in
            self.handleAppRefresh(task: task as! BGAppRefreshTask)
        }
    }

    func scheduleAppRefresh() {
        let request = BGAppRefreshTaskRequest(identifier: Self.refreshTaskID)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 15 * 60) // 15분 후

        do {
            try BGTaskScheduler.shared.submit(request)
            print("✅ App refresh scheduled")
        } catch {
            print("❌ Failed to schedule: \(error)")
        }
    }

    private func handleAppRefresh(task: BGAppRefreshTask) {
        scheduleAppRefresh() // 다음 실행 예약

        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1

        let operation = RefreshOperation()

        // 시간 초과 핸들러
        task.expirationHandler = {
            queue.cancelAllOperations()
        }

        operation.completionBlock = {
            task.setTaskCompleted(success: !operation.isCancelled)
        }

        queue.addOperation(operation)
    }
}

class RefreshOperation: Operation {
    override func main() {
        guard !isCancelled else { return }

        // 가벼운 작업 수행
        // 예: API에서 최신 데이터 가져오기
        let semaphore = DispatchSemaphore(value: 0)

        URLSession.shared.dataTask(with: URL(string: "https://api.example.com/data")!) { data, response, error in
            if let data = data {
                // 데이터 처리
                print("Data updated: \(data.count) bytes")
            }
            semaphore.signal()
        }.resume()

        semaphore.wait()
    }
}
```

### 2. Background Processing Task (BGProcessingTask)

**특징:**
- 실행 시간: 수 분 (1-10분 정도)
- 용도: 무거운 작업 (ML 학습, 대량 데이터 처리)
- 실행 조건: 충전 중, WiFi 연결, 배터리 충분
- 빈도: 하루 1-2회 정도

**구현 예제:**

```swift
class BackgroundTaskManager {
    static let processingTaskID = "com.yourapp.processing"

    func registerProcessingTask() {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: Self.processingTaskID,
            using: nil
        ) { task in
            self.handleProcessing(task: task as! BGProcessingTask)
        }
    }

    func scheduleProcessing() {
        let request = BGProcessingTaskRequest(identifier: Self.processingTaskID)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 60 * 60) // 1시간 후
        request.requiresNetworkConnectivity = true
        request.requiresExternalPower = true // 충전 중일 때만

        do {
            try BGTaskScheduler.shared.submit(request)
            print("✅ Processing task scheduled")
        } catch {
            print("❌ Failed to schedule: \(error)")
        }
    }

    private func handleProcessing(task: BGProcessingTask) {
        scheduleProcessing()

        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1

        let operation = ProcessingOperation()

        task.expirationHandler = {
            queue.cancelAllOperations()
        }

        operation.completionBlock = {
            task.setTaskCompleted(success: !operation.isCancelled)
        }

        queue.addOperation(operation)
    }
}

class ProcessingOperation: Operation {
    override func main() {
        guard !isCancelled else { return }

        // 무거운 작업 수행
        print("🔄 Processing heavy task...")

        // 예: 대량 데이터 처리
        for i in 0..<1000 {
            guard !isCancelled else {
                print("⚠️ Task cancelled at iteration \(i)")
                return
            }

            // 처리 로직
            Thread.sleep(forTimeInterval: 0.01)
        }

        print("✅ Processing completed")
    }
}
```

### 3. URLSession Background Transfer

**특징:**
- 앱이 종료되어도 다운로드/업로드 계속 진행
- 완료 시 앱을 깨워서 알림
- 대용량 파일 전송에 최적

**구현 예제:**

```swift
class BackgroundUploadService: NSObject {
    static let shared = BackgroundUploadService()

    private var session: URLSession!
    private var completionHandlers: [String: (Result<Data, Error>) -> Void] = [:]

    override init() {
        super.init()

        let config = URLSessionConfiguration.background(
            withIdentifier: "com.yourapp.background.upload"
        )
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true

        session = URLSession(
            configuration: config,
            delegate: self,
            delegateQueue: nil
        )
    }

    func uploadFile(
        fileURL: URL,
        to serverURL: URL,
        completion: @escaping (Result<Data, Error>) -> Void
    ) {
        var request = URLRequest(url: serverURL)
        request.httpMethod = "POST"
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")

        let task = session.uploadTask(with: request, fromFile: fileURL)
        completionHandlers[task.taskIdentifier.description] = completion
        task.resume()

        print("📤 Upload started: \(fileURL.lastPathComponent)")
    }
}

extension BackgroundUploadService: URLSessionDelegate, URLSessionTaskDelegate, URLSessionDataDelegate {
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        let taskID = task.taskIdentifier.description

        if let error = error {
            completionHandlers[taskID]?(.failure(error))
            print("❌ Upload failed: \(error)")
        } else {
            print("✅ Upload completed")
        }

        completionHandlers.removeValue(forKey: taskID)
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive data: Data
    ) {
        let taskID = dataTask.taskIdentifier.description
        completionHandlers[taskID]?(.success(data))
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        DispatchQueue.main.async {
            guard let appDelegate = UIApplication.shared.delegate as? AppDelegate,
                  let completionHandler = appDelegate.backgroundCompletionHandler else {
                return
            }
            completionHandler()
        }
    }
}
```

### 4. Background Task Completion (Legacy)

**특징:**
- iOS 13 이전부터 사용
- foreground → background 전환 시 실행
- 실행 시간: 약 30초
- 간단한 마무리 작업에 사용

**구현 예제:**

```swift
class AppDelegate: UIResponder, UIApplicationDelegate {
    func applicationDidEnterBackground(_ application: UIApplication) {
        var backgroundTask: UIBackgroundTaskIdentifier = .invalid

        backgroundTask = application.beginBackgroundTask {
            // 시간 초과 시 호출
            print("⚠️ Background task expired")
            application.endBackgroundTask(backgroundTask)
            backgroundTask = .invalid
        }

        // 작업 수행
        DispatchQueue.global().async {
            // 마무리 작업 (데이터 저장, 로그 전송 등)
            print("🔄 Finishing up...")
            Thread.sleep(forTimeInterval: 5)

            // 작업 완료
            print("✅ Background task completed")
            application.endBackgroundTask(backgroundTask)
            backgroundTask = .invalid
        }
    }
}
```

### 5. Silent Push Notifications

**특징:**
- 서버에서 트리거
- 실행 시간: 약 30초
- 사용자에게 알림 표시 안함
- 특정 이벤트 발생 시 앱 깨우기

**구현 예제:**

```swift
class AppDelegate: UIResponder, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        // Push notification 등록
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) { granted, error in
            if granted {
                DispatchQueue.main.async {
                    application.registerForRemoteNotifications()
                }
            }
        }
        return true
    }

    func application(
        _ application: UIApplication,
        didReceiveRemoteNotification userInfo: [AnyHashable: Any],
        fetchCompletionHandler completionHandler: @escaping (UIBackgroundFetchResult) -> Void
    ) {
        // Silent push 처리
        print("📩 Silent push received")

        // 백그라운드 작업 수행
        if let data = userInfo["data"] as? [String: Any] {
            // 데이터 처리
            print("Processing data: \(data)")

            // 작업 완료 알림
            completionHandler(.newData)
        } else {
            completionHandler(.noData)
        }
    }
}

// Silent Push Payload (JSON)
// {
//   "aps": {
//     "content-available": 1
//   },
//   "data": {
//     "action": "sync",
//     "timestamp": 1234567890
//   }
// }
```

## AppDelegate 통합 설정

```swift
import UIKit
import BackgroundTasks

@main
class AppDelegate: UIResponder, UIApplicationDelegate {

    var backgroundCompletionHandler: (() -> Void)?

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        // Background Tasks 등록
        BackgroundTaskManager.shared.registerBackgroundTasks()
        BackgroundTaskManager.shared.registerProcessingTask()

        return true
    }

    func applicationDidEnterBackground(_ application: UIApplication) {
        // Background Tasks 스케줄링
        BackgroundTaskManager.shared.scheduleAppRefresh()
        BackgroundTaskManager.shared.scheduleProcessing()
    }

    // Background URLSession 완료 핸들러
    func application(
        _ application: UIApplication,
        handleEventsForBackgroundURLSession identifier: String,
        completionHandler: @escaping () -> Void
    ) {
        backgroundCompletionHandler = completionHandler
    }
}
```

## Info.plist 설정

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- Background Task Identifiers -->
    <key>BGTaskSchedulerPermittedIdentifiers</key>
    <array>
        <string>com.yourapp.refresh</string>
        <string>com.yourapp.processing</string>
    </array>

    <!-- Background Modes -->
    <key>UIBackgroundModes</key>
    <array>
        <string>fetch</string>
        <string>processing</string>
        <string>remote-notification</string>
    </array>

    <!-- Permissions -->
    <key>NSPhotoLibraryUsageDescription</key>
    <string>We need access to your photos for processing</string>
</dict>
</plist>
```

# 실전 예제: 대량 사진 Moderation & Upload

5,000장의 사진을 CoreML로 필터링하고 서버에 업로드하는 완전한 구현 예제입니다.

## 1. CoreML Moderation Model Service

```swift
import CoreML
import Vision
import UIKit

class ModerationService {
    static let shared = ModerationService()

    private var model: VNCoreMLModel?
    private let processingQueue = DispatchQueue(
        label: "com.yourapp.moderation",
        qos: .userInitiated
    )

    private init() {
        setupModel()
    }

    private func setupModel() {
        guard let modelURL = Bundle.main.url(
            forResource: "ModerationModel",
            withExtension: "mlmodelc"
        ) else {
            print("❌ Model file not found")
            return
        }

        do {
            let mlModel = try MLModel(contentsOf: modelURL)
            model = try VNCoreMLModel(for: mlModel)
            print("✅ CoreML model loaded")
        } catch {
            print("❌ Failed to load model: \(error)")
        }
    }

    func analyzeImage(
        _ image: UIImage,
        completion: @escaping (ModerationResult) -> Void
    ) {
        guard let model = model,
              let cgImage = image.cgImage else {
            completion(.init(isAppropriate: false, confidence: 0, categories: []))
            return
        }

        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNClassificationObservation] else {
                completion(.init(isAppropriate: false, confidence: 0, categories: []))
                return
            }

            // 상위 3개 결과 추출
            let topResults = results.prefix(3).map { observation in
                CategoryScore(
                    category: observation.identifier,
                    confidence: observation.confidence
                )
            }

            let isAppropriate = results.first?.identifier == "appropriate"
            let confidence = results.first?.confidence ?? 0

            completion(.init(
                isAppropriate: isAppropriate,
                confidence: confidence,
                categories: topResults
            ))
        }

        let handler = VNImageRequestHandler(
            cgImage: cgImage,
            options: [:]
        )

        processingQueue.async {
            do {
                try handler.perform([request])
            } catch {
                print("❌ Failed to perform request: \(error)")
                completion(.init(isAppropriate: false, confidence: 0, categories: []))
            }
        }
    }

    // Batch processing for efficiency
    func analyzeBatch(
        _ images: [UIImage],
        progressHandler: @escaping (Int, Int) -> Void,
        completion: @escaping ([ModerationResult]) -> Void
    ) {
        var results: [ModerationResult] = []
        let group = DispatchGroup()
        let lock = NSLock()

        for (index, image) in images.enumerated() {
            group.enter()

            analyzeImage(image) { result in
                lock.lock()
                results.append(result)
                let currentCount = results.count
                lock.unlock()

                progressHandler(currentCount, images.count)
                group.leave()
            }
        }

        group.notify(queue: .main) {
            completion(results)
        }
    }
}

struct ModerationResult {
    let isAppropriate: Bool
    let confidence: Float
    let categories: [CategoryScore]
}

struct CategoryScore {
    let category: String
    let confidence: Float
}
```

## 2. Photo Processing Service

```swift
import Photos
import UIKit

class PhotoProcessingService {
    static let shared = PhotoProcessingService()

    private let processingQueue = DispatchQueue(
        label: "com.yourapp.photoprocessing",
        qos: .utility,
        attributes: .concurrent
    )

    private let moderationService = ModerationService.shared
    private let uploadService = UploadService.shared
    private let cache = ProcessingCache.shared

    private init() {}

    func processAllPhotos(
        progressHandler: @escaping (ProcessingProgress) -> Void,
        completion: @escaping (ProcessingResult) -> Void
    ) {
        // Photos 권한 확인
        PHPhotoLibrary.requestAuthorization { status in
            guard status == .authorized else {
                print("❌ Photo library access denied")
                completion(.init(
                    success: false,
                    processedCount: 0,
                    uploadedCount: 0,
                    skippedCount: 0,
                    failedCount: 0
                ))
                return
            }

            self.fetchAndProcessPhotos(
                progressHandler: progressHandler,
                completion: completion
            )
        }
    }

    private func fetchAndProcessPhotos(
        progressHandler: @escaping (ProcessingProgress) -> Void,
        completion: @escaping (ProcessingResult) -> Void
    ) {
        let fetchOptions = PHFetchOptions()
        fetchOptions.sortDescriptors = [
            NSSortDescriptor(key: "creationDate", ascending: false)
        ]

        let allPhotos = PHAsset.fetchAssets(with: .image, options: fetchOptions)
        let totalCount = allPhotos.count

        print("📸 Found \(totalCount) photos to process")

        // 이미 처리된 사진 로드
        let processedAssets = cache.getProcessedAssets()

        var stats = ProcessingStats()
        stats.total = totalCount
        let lock = NSLock()

        let group = DispatchGroup()
        let semaphore = DispatchSemaphore(value: 5) // 동시 5개 처리

        allPhotos.enumerateObjects { asset, index, stop in
            // 이미 처리된 사진은 스킵
            if processedAssets.contains(asset.localIdentifier) {
                lock.lock()
                stats.skipped += 1
                lock.unlock()

                progressHandler(stats.toProgress())
                return
            }

            group.enter()
            semaphore.wait()

            self.processingQueue.async {
                self.processPhoto(asset) { result in
                    lock.lock()
                    stats.processed += 1

                    switch result {
                    case .uploaded:
                        stats.uploaded += 1
                        self.cache.markAsProcessed(asset.localIdentifier)
                    case .filtered:
                        stats.filtered += 1
                        self.cache.markAsProcessed(asset.localIdentifier)
                    case .failed:
                        stats.failed += 1
                    }

                    let progress = stats.toProgress()
                    lock.unlock()

                    DispatchQueue.main.async {
                        progressHandler(progress)
                    }

                    semaphore.signal()
                    group.leave()
                }
            }
        }

        group.notify(queue: .main) {
            let result = ProcessingResult(
                success: true,
                processedCount: stats.processed,
                uploadedCount: stats.uploaded,
                skippedCount: stats.skipped,
                failedCount: stats.failed
            )

            print("✅ Processing completed: \(result)")
            completion(result)
        }
    }

    private func processPhoto(
        _ asset: PHAsset,
        completion: @escaping (PhotoProcessingResult) -> Void
    ) {
        let options = PHImageRequestOptions()
        options.deliveryMode = .highQualityFormat
        options.isNetworkAccessAllowed = true
        options.isSynchronous = false

        // 적절한 크기로 요청 (메모리 절약)
        let targetSize = CGSize(width: 1024, height: 1024)

        PHImageManager.default().requestImage(
            for: asset,
            targetSize: targetSize,
            contentMode: .aspectFit,
            options: options
        ) { image, info in
            guard let image = image else {
                print("❌ Failed to load image: \(asset.localIdentifier)")
                completion(.failed)
                return
            }

            // CoreML Moderation 실행
            self.moderationService.analyzeImage(image) { result in
                print("🔍 Moderation result: \(result.isAppropriate ? "✅" : "⚠️") confidence: \(result.confidence)")

                // 적절하지 않은 이미지는 필터링
                guard result.isAppropriate && result.confidence > 0.8 else {
                    print("🚫 Image filtered: \(asset.localIdentifier)")
                    completion(.filtered)
                    return
                }

                // 적절한 이미지만 업로드
                self.uploadService.uploadImage(
                    image,
                    assetId: asset.localIdentifier,
                    metadata: ImageMetadata(
                        creationDate: asset.creationDate,
                        location: asset.location,
                        moderationScore: result.confidence
                    )
                ) { success in
                    if success {
                        print("✅ Image uploaded: \(asset.localIdentifier)")
                        completion(.uploaded)
                    } else {
                        print("❌ Upload failed: \(asset.localIdentifier)")
                        completion(.failed)
                    }
                }
            }
        }
    }
}

// MARK: - Supporting Types

enum PhotoProcessingResult {
    case uploaded
    case filtered
    case failed
}

struct ProcessingStats {
    var total: Int = 0
    var processed: Int = 0
    var uploaded: Int = 0
    var filtered: Int = 0
    var skipped: Int = 0
    var failed: Int = 0

    func toProgress() -> ProcessingProgress {
        ProcessingProgress(
            total: total,
            processed: processed,
            uploaded: uploaded,
            filtered: filtered,
            skipped: skipped,
            failed: failed
        )
    }
}

struct ProcessingProgress {
    let total: Int
    let processed: Int
    let uploaded: Int
    let filtered: Int
    let skipped: Int
    let failed: Int

    var percentage: Double {
        guard total > 0 else { return 0 }
        return Double(processed + skipped) / Double(total) * 100
    }
}

struct ProcessingResult: CustomStringConvertible {
    let success: Bool
    let processedCount: Int
    let uploadedCount: Int
    let skippedCount: Int
    let failedCount: Int

    var description: String {
        """
        ProcessingResult(
            processed: \(processedCount),
            uploaded: \(uploadedCount),
            skipped: \(skippedCount),
            failed: \(failedCount)
        )
        """
    }
}

struct ImageMetadata {
    let creationDate: Date?
    let location: CLLocation?
    let moderationScore: Float
}
```

## 3. Upload Service (Background URLSession)

```swift
import Foundation

class UploadService: NSObject {
    static let shared = UploadService()

    private var session: URLSession!
    private var uploadCompletions: [String: (Bool) -> Void] = [:]
    private let lock = NSLock()

    private override init() {
        super.init()

        let config = URLSessionConfiguration.background(
            withIdentifier: "com.yourapp.photo.upload"
        )
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        config.timeoutIntervalForRequest = 300 // 5분
        config.timeoutIntervalForResource = 3600 // 1시간

        session = URLSession(
            configuration: config,
            delegate: self,
            delegateQueue: nil
        )
    }

    func uploadImage(
        _ image: UIImage,
        assetId: String,
        metadata: ImageMetadata,
        completion: @escaping (Bool) -> Void
    ) {
        // JPEG 압축 (품질 80%)
        guard let imageData = image.jpegData(compressionQuality: 0.8) else {
            print("❌ Failed to convert image to JPEG")
            completion(false)
            return
        }

        // 임시 파일 저장
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent("\(assetId).jpg")

        do {
            try imageData.write(to: fileURL)
        } catch {
            print("❌ Failed to write temp file: \(error)")
            completion(false)
            return
        }

        // 서버 URL 생성
        guard let url = URL(string: "https://your-api.com/api/v1/photos/upload") else {
            completion(false)
            return
        }

        // Multipart form data 생성
        var request = URLRequest(url: url)
        request.httpMethod = "POST"

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer YOUR_AUTH_TOKEN", forHTTPHeaderField: "Authorization")

        // Multipart body 생성
        let bodyURL = createMultipartBody(
            imageData: imageData,
            assetId: assetId,
            metadata: metadata,
            boundary: boundary
        )

        // Upload Task 생성
        let task = session.uploadTask(with: request, fromFile: bodyURL)

        lock.lock()
        uploadCompletions[task.taskIdentifier.description] = completion
        lock.unlock()

        task.resume()

        print("📤 Upload started: \(assetId)")
    }

    private func createMultipartBody(
        imageData: Data,
        assetId: String,
        metadata: ImageMetadata,
        boundary: String
    ) -> URL {
        var body = Data()

        // Image data
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(assetId).jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n".data(using: .utf8)!)

        // Metadata
        if let metadataJSON = try? JSONEncoder().encode(metadata),
           let metadataString = String(data: metadataJSON, encoding: .utf8) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"metadata\"\r\n\r\n".data(using: .utf8)!)
            body.append(metadataString.data(using: .utf8)!)
            body.append("\r\n".data(using: .utf8)!)
        }

        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        // 임시 파일에 저장
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("upload-\(UUID().uuidString).dat")
        try? body.write(to: tempURL)

        return tempURL
    }
}

// MARK: - URLSession Delegate

extension UploadService: URLSessionDelegate, URLSessionTaskDelegate, URLSessionDataDelegate {
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        let taskID = task.taskIdentifier.description

        lock.lock()
        let completion = uploadCompletions[taskID]
        uploadCompletions.removeValue(forKey: taskID)
        lock.unlock()

        if let error = error {
            print("❌ Upload failed: \(error.localizedDescription)")
            completion?(false)
        } else if let httpResponse = task.response as? HTTPURLResponse {
            let success = (200...299).contains(httpResponse.statusCode)
            print(success ? "✅ Upload succeeded" : "❌ Upload failed: HTTP \(httpResponse.statusCode)")
            completion?(success)
        } else {
            completion?(false)
        }

        // 임시 파일 삭제
        if let fileURL = (task as? URLSessionUploadTask)?.currentRequest?.url {
            try? FileManager.default.removeItem(at: fileURL)
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didSendBodyData bytesSent: Int64,
        totalBytesSent: Int64,
        totalBytesExpectedToSend: Int64
    ) {
        let progress = Double(totalBytesSent) / Double(totalBytesExpectedToSend) * 100
        print("📊 Upload progress: \(String(format: "%.1f", progress))%")
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        DispatchQueue.main.async {
            guard let appDelegate = UIApplication.shared.delegate as? AppDelegate,
                  let completionHandler = appDelegate.backgroundCompletionHandler else {
                return
            }

            print("✅ Background URLSession finished")
            completionHandler()
        }
    }
}
```

## 4. Processing Cache (진행 상태 저장)

```swift
import Foundation

class ProcessingCache {
    static let shared = ProcessingCache()

    private let processedKey = "processedAssets"
    private let defaults = UserDefaults.standard
    private let lock = NSLock()

    private init() {}

    func markAsProcessed(_ assetId: String) {
        lock.lock()
        defer { lock.unlock() }

        var processed = getProcessedAssets()
        processed.insert(assetId)
        defaults.set(Array(processed), forKey: processedKey)
    }

    func getProcessedAssets() -> Set<String> {
        lock.lock()
        defer { lock.unlock() }

        let array = defaults.array(forKey: processedKey) as? [String] ?? []
        return Set(array)
    }

    func clearProcessedAssets() {
        lock.lock()
        defer { lock.unlock() }

        defaults.removeObject(forKey: processedKey)
    }

    func getProcessedCount() -> Int {
        return getProcessedAssets().count
    }
}
```

## 5. Background Task Integration

```swift
import BackgroundTasks

class PhotoModerationBackgroundTask {
    static let identifier = "com.yourapp.photomoderation"

    static func register() {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: identifier,
            using: nil
        ) { task in
            handlePhotoModeration(task: task as! BGProcessingTask)
        }
    }

    static func schedule() {
        let request = BGProcessingTaskRequest(identifier: identifier)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 2 * 60 * 60) // 2시간 후
        request.requiresNetworkConnectivity = true
        request.requiresExternalPower = true

        do {
            try BGTaskScheduler.shared.submit(request)
            print("✅ Photo moderation task scheduled")
        } catch {
            print("❌ Failed to schedule: \(error)")
        }
    }

    private static func handlePhotoModeration(task: BGProcessingTask) {
        schedule() // 다음 실행 예약

        var processingCompleted = false

        // 시간 초과 핸들러
        task.expirationHandler = {
            print("⚠️ Task will expire soon")
            if !processingCompleted {
                task.setTaskCompleted(success: false)
            }
        }

        // 사진 처리 시작
        PhotoProcessingService.shared.processAllPhotos(
            progressHandler: { progress in
                print("""
                📊 Progress: \(String(format: "%.1f", progress.percentage))%
                   Processed: \(progress.processed)/\(progress.total)
                   Uploaded: \(progress.uploaded)
                   Filtered: \(progress.filtered)
                   Skipped: \(progress.skipped)
                   Failed: \(progress.failed)
                """)
            },
            completion: { result in
                processingCompleted = true
                print("""
                ✅ Photo moderation completed
                   Processed: \(result.processedCount)
                   Uploaded: \(result.uploadedCount)
                   Skipped: \(result.skippedCount)
                   Failed: \(result.failedCount)
                """)

                task.setTaskCompleted(success: result.success)
            }
        )
    }
}
```

## 6. UI Integration (SwiftUI)

```swift
import SwiftUI

struct PhotoModerationView: View {
    @StateObject private var viewModel = PhotoModerationViewModel()

    var body: some View {
        VStack(spacing: 20) {
            Text("Photo Moderation")
                .font(.largeTitle)
                .bold()

            if viewModel.isProcessing {
                VStack(spacing: 15) {
                    ProgressView(value: viewModel.progress.percentage, total: 100)
                        .progressViewStyle(.linear)

                    Text("\(String(format: "%.1f", viewModel.progress.percentage))%")
                        .font(.headline)

                    VStack(alignment: .leading, spacing: 8) {
                        ProgressRow(label: "Processed", value: viewModel.progress.processed, total: viewModel.progress.total)
                        ProgressRow(label: "Uploaded", value: viewModel.progress.uploaded, total: viewModel.progress.total)
                        ProgressRow(label: "Filtered", value: viewModel.progress.filtered, total: viewModel.progress.total)
                        ProgressRow(label: "Skipped", value: viewModel.progress.skipped, total: viewModel.progress.total)
                        ProgressRow(label: "Failed", value: viewModel.progress.failed, total: viewModel.progress.total)
                    }
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(10)
                }
                .padding()
            }

            Button(action: {
                viewModel.startProcessing()
            }) {
                Text(viewModel.isProcessing ? "Processing..." : "Start Processing")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(viewModel.isProcessing ? Color.gray : Color.blue)
                    .cornerRadius(10)
            }
            .disabled(viewModel.isProcessing)
            .padding()

            if let result = viewModel.result {
                VStack(alignment: .leading, spacing: 8) {
                    Text("✅ Completed")
                        .font(.headline)
                        .foregroundColor(.green)

                    Text("Uploaded: \(result.uploadedCount)")
                    Text("Processed: \(result.processedCount)")
                    Text("Skipped: \(result.skippedCount)")
                    Text("Failed: \(result.failedCount)")
                }
                .padding()
                .background(Color.green.opacity(0.1))
                .cornerRadius(10)
            }

            Spacer()
        }
        .padding()
    }
}

struct ProgressRow: View {
    let label: String
    let value: Int
    let total: Int

    var body: some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundColor(.secondary)

            Spacer()

            Text("\(value)/\(total)")
                .font(.subheadline)
                .bold()
        }
    }
}

class PhotoModerationViewModel: ObservableObject {
    @Published var isProcessing = false
    @Published var progress = ProcessingProgress(total: 0, processed: 0, uploaded: 0, filtered: 0, skipped: 0, failed: 0)
    @Published var result: ProcessingResult?

    func startProcessing() {
        isProcessing = true
        result = nil

        PhotoProcessingService.shared.processAllPhotos(
            progressHandler: { [weak self] progress in
                DispatchQueue.main.async {
                    self?.progress = progress
                }
            },
            completion: { [weak self] result in
                DispatchQueue.main.async {
                    self?.isProcessing = false
                    self?.result = result
                }
            }
        )
    }
}
```

# iOS 18 vs iOS 26 GPU 할당

## iOS 18: CPU 중심 처리

iOS 18에서는 백그라운드에서 GPU 사용이 매우 제한적입니다.

### 제약사항
- 백그라운드 GPU 우선순위 낮음
- Metal 연산 중단 가능
- CoreML GPU 추론 제한적

### iOS 18 최적화 전략

```swift
import CoreML

class ModerationServiceIOS18 {
    static let shared = ModerationServiceIOS18()

    private var model: MLModel?

    private init() {
        setupModel()
    }

    private func setupModel() {
        let config = MLModelConfiguration()

        // iOS 18: CPU만 사용
        config.computeUnits = .cpuOnly

        // 또는 혼합 모드 (백그라운드에서는 주로 CPU)
        // config.computeUnits = .cpuAndGPU

        guard let modelURL = Bundle.main.url(
            forResource: "ModerationModel",
            withExtension: "mlmodelc"
        ) else {
            print("❌ Model not found")
            return
        }

        do {
            model = try MLModel(contentsOf: modelURL, configuration: config)
            print("✅ Model loaded with CPU compute units")
        } catch {
            print("❌ Failed to load model: \(error)")
        }
    }

    func predict(pixelBuffer: CVPixelBuffer) throws -> ModerationOutput {
        guard let model = model else {
            throw ModerationError.modelNotLoaded
        }

        let input = ModerationInput(image: pixelBuffer)
        let prediction = try model.prediction(from: input)

        return ModerationOutput(from: prediction)
    }
}

// Batch Processing for CPU Efficiency
extension ModerationServiceIOS18 {
    func predictBatch(_ pixelBuffers: [CVPixelBuffer]) throws -> [ModerationOutput] {
        guard let model = model else {
            throw ModerationError.modelNotLoaded
        }

        var results: [ModerationOutput] = []

        // CPU에서 배치 처리하여 효율성 향상
        for pixelBuffer in pixelBuffers {
            let input = ModerationInput(image: pixelBuffer)
            let prediction = try model.prediction(from: input)
            results.append(ModerationOutput(from: prediction))
        }

        return results
    }
}

enum ModerationError: Error {
    case modelNotLoaded
    case predictionFailed
}

struct ModerationInput: MLFeatureProvider {
    var image: CVPixelBuffer

    var featureNames: Set<String> {
        return ["image"]
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "image" {
            return MLFeatureValue(pixelBuffer: image)
        }
        return nil
    }

    init(image: CVPixelBuffer) {
        self.image = image
    }
}

struct ModerationOutput {
    let category: String
    let confidence: Float

    init(from prediction: MLFeatureProvider) {
        self.category = prediction.featureValue(for: "classLabel")?.stringValue ?? "unknown"
        self.confidence = prediction.featureValue(for: "confidence")?.floatValue ?? 0.0
    }
}
```

### iOS 18 성능 최적화

```swift
class OptimizedPhotoProcessingIOS18 {
    static let shared = OptimizedPhotoProcessingIOS18()

    // CPU 처리를 위한 설정
    private let maxConcurrent = 3 // CPU 부하 관리
    private let batchSize = 10 // 배치 크기

    func processPhotos() {
        let semaphore = DispatchSemaphore(value: maxConcurrent)

        // ... 사진 가져오기 ...

        // 배치 단위로 처리
        let batches = photos.chunked(into: batchSize)

        for batch in batches {
            semaphore.wait()

            processingQueue.async {
                self.processBatch(batch) {
                    semaphore.signal()
                }
            }
        }
    }

    private func processBatch(_ photos: [PHAsset], completion: @escaping () -> Void) {
        // 배치 처리로 오버헤드 감소
        // ...
        completion()
    }
}

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}
```

## iOS 26: GPU 백그라운드 지원

iOS 26부터는 백그라운드에서 GPU를 활용할 수 있습니다.

### 새로운 기능
- 백그라운드 GPU 우선순위 향상
- Metal Performance Shaders 지원
- CoreML GPU 추론 최적화
- 전용 GPU 메모리 할당

### iOS 26 구현

```swift
import CoreML
import MetalPerformanceShaders

@available(iOS 26.0, *)
class ModerationServiceIOS26 {
    static let shared = ModerationServiceIOS26()

    private var model: MLModel?
    private var metalDevice: MTLDevice?
    private var gpuAvailable: Bool = false

    private init() {
        setupMetal()
        setupModel()
    }

    private func setupMetal() {
        metalDevice = MTLCreateSystemDefaultDevice()
        gpuAvailable = metalDevice != nil

        if gpuAvailable {
            print("✅ GPU available for background processing")
        }
    }

    private func setupModel() {
        let config = MLModelConfiguration()

        if #available(iOS 26.0, *) {
            // iOS 26: GPU 사용 가능
            config.computeUnits = .all

            // 백그라운드 GPU 명시적 허용
            config.allowsBackgroundGPUCompute = true

            // GPU 메모리 제한 설정 (옵션)
            config.gpuMemoryLimit = 2 * 1024 * 1024 * 1024 // 2GB

            // GPU 우선순위 설정
            config.gpuPriority = .high
        } else {
            config.computeUnits = .cpuOnly
        }

        guard let modelURL = Bundle.main.url(
            forResource: "ModerationModel",
            withExtension: "mlmodelc"
        ) else {
            print("❌ Model not found")
            return
        }

        do {
            model = try MLModel(contentsOf: modelURL, configuration: config)
            print("✅ Model loaded with GPU support")
        } catch {
            print("❌ Failed to load model: \(error)")
        }
    }

    func predict(pixelBuffer: CVPixelBuffer) async throws -> ModerationOutput {
        guard let model = model else {
            throw ModerationError.modelNotLoaded
        }

        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let input = ModerationInput(image: pixelBuffer)
                    let prediction = try model.prediction(from: input)
                    continuation.resume(returning: ModerationOutput(from: prediction))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    // GPU 가속 배치 처리
    func predictBatch(_ pixelBuffers: [CVPixelBuffer]) async throws -> [ModerationOutput] {
        guard let model = model else {
            throw ModerationError.modelNotLoaded
        }

        // GPU에서 병렬 처리
        return try await withThrowingTaskGroup(of: ModerationOutput.self) { group in
            for pixelBuffer in pixelBuffers {
                group.addTask {
                    let input = ModerationInput(image: pixelBuffer)
                    let prediction = try model.prediction(from: input)
                    return ModerationOutput(from: prediction)
                }
            }

            var results: [ModerationOutput] = []
            for try await result in group {
                results.append(result)
            }
            return results
        }
    }
}

// Metal Performance Shaders를 활용한 전처리
@available(iOS 26.0, *)
extension ModerationServiceIOS26 {
    func preprocessWithMPS(cgImage: CGImage) -> CVPixelBuffer? {
        guard let device = metalDevice else { return nil }

        let textureLoader = MTKTextureLoader(device: device)

        do {
            // CGImage를 Metal Texture로 변환
            let texture = try textureLoader.newTexture(
                cgImage: cgImage,
                options: [
                    .textureUsage: MTLTextureUsage.shaderRead.rawValue,
                    .SRGB: false
                ]
            )

            // Lanczos scaling으로 고품질 리사이징
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba8Unorm,
                width: 224,
                height: 224,
                mipmapped: false
            )
            descriptor.usage = [.shaderRead, .shaderWrite]

            guard let resizedTexture = device.makeTexture(descriptor: descriptor) else {
                return nil
            }

            // Command buffer 생성
            guard let commandQueue = device.makeCommandQueue(),
                  let commandBuffer = commandQueue.makeCommandBuffer() else {
                return nil
            }

            // Lanczos scaling 적용
            let scaler = MPSImageLanczosScale(device: device)
            scaler.encode(
                commandBuffer: commandBuffer,
                sourceTexture: texture,
                destinationTexture: resizedTexture
            )

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            // Texture를 CVPixelBuffer로 변환
            return convertTextureToPixelBuffer(texture: resizedTexture)
        } catch {
            print("❌ MPS preprocessing failed: \(error)")
            return nil
        }
    }

    private func convertTextureToPixelBuffer(texture: MTLTexture) -> CVPixelBuffer? {
        let width = texture.width
        let height = texture.height

        var pixelBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferMetalCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let baseAddress = CVPixelBufferGetBaseAddress(buffer)

        texture.getBytes(
            baseAddress!,
            bytesPerRow: bytesPerRow,
            from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                           size: MTLSize(width: width, height: height, depth: 1)),
            mipmapLevel: 0
        )

        return buffer
    }
}
```

### iOS 26 최적화된 Photo Processing

```swift
@available(iOS 26.0, *)
class OptimizedPhotoProcessingIOS26 {
    static let shared = OptimizedPhotoProcessingIOS26()

    // GPU 처리를 위한 설정
    private let maxConcurrent = 20 // GPU로 더 많은 동시 처리 가능
    private let batchSize = 50 // 더 큰 배치 크기

    func processPhotos() async {
        await withTaskGroup(of: Void.self) { group in
            let batches = photos.chunked(into: batchSize)

            for batch in batches {
                group.addTask {
                    await self.processBatchAsync(batch)
                }
            }
        }
    }

    private func processBatchAsync(_ photos: [PHAsset]) async {
        // GPU 가속 배치 처리
        let pixelBuffers = await loadPixelBuffers(photos)

        do {
            let results = try await ModerationServiceIOS26.shared.predictBatch(pixelBuffers)

            // 결과 처리
            for (index, result) in results.enumerated() {
                if result.confidence > 0.8 {
                    await uploadPhoto(photos[index])
                }
            }
        } catch {
            print("❌ Batch processing failed: \(error)")
        }
    }

    private func loadPixelBuffers(_ photos: [PHAsset]) async -> [CVPixelBuffer] {
        await withTaskGroup(of: CVPixelBuffer?.self) { group in
            for photo in photos {
                group.addTask {
                    return await self.loadPixelBuffer(photo)
                }
            }

            var buffers: [CVPixelBuffer] = []
            for await buffer in group {
                if let buffer = buffer {
                    buffers.append(buffer)
                }
            }
            return buffers
        }
    }

    private func loadPixelBuffer(_ photo: PHAsset) async -> CVPixelBuffer? {
        // ... 구현 ...
        return nil
    }

    private func uploadPhoto(_ photo: PHAsset) async {
        // ... 구현 ...
    }
}
```

## 버전별 분기 처리

```swift
class AdaptiveModerationService {
    static let shared = AdaptiveModerationService()

    private init() {}

    func predict(pixelBuffer: CVPixelBuffer) async throws -> ModerationOutput {
        if #available(iOS 26.0, *) {
            // iOS 26: GPU 가속 사용
            return try await ModerationServiceIOS26.shared.predict(pixelBuffer: pixelBuffer)
        } else {
            // iOS 18: CPU 사용
            return try ModerationServiceIOS18.shared.predict(pixelBuffer: pixelBuffer)
        }
    }

    func predictBatch(_ pixelBuffers: [CVPixelBuffer]) async throws -> [ModerationOutput] {
        if #available(iOS 26.0, *) {
            // iOS 26: GPU 병렬 처리
            return try await ModerationServiceIOS26.shared.predictBatch(pixelBuffers)
        } else {
            // iOS 18: CPU 순차 처리
            return try ModerationServiceIOS18.shared.predictBatch(pixelBuffers)
        }
    }
}
```

## System Monitor (배터리 & 열 관리)

```swift
import UIKit

class SystemMonitor {
    static let shared = SystemMonitor()

    private init() {
        UIDevice.current.isBatteryMonitoringEnabled = true
    }

    func canProcessInBackground() -> Bool {
        let device = UIDevice.current

        // 배터리 확인
        let batteryLevel = device.batteryLevel
        let batteryState = device.batteryState
        let batteryOK = batteryLevel > 0.2 || batteryState == .charging || batteryState == .full

        // 열 상태 확인
        let thermalState = ProcessInfo.processInfo.thermalState
        let thermalOK = thermalState != .critical && thermalState != .serious

        // 메모리 확인
        let memoryOK = hasEnoughMemory()

        let canProcess = batteryOK && thermalOK && memoryOK

        print("""
        📊 System Status:
           Battery: \(batteryLevel * 100)% (\(batteryState.description))
           Thermal: \(thermalState.description)
           Memory: \(memoryOK ? "✅" : "⚠️")
           Can Process: \(canProcess ? "✅" : "❌")
        """)

        return canProcess
    }

    private func hasEnoughMemory() -> Bool {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4

        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        guard kerr == KERN_SUCCESS else { return false }

        let usedMemory = Double(info.resident_size) / 1024 / 1024 // MB
        let availableMemory = Double(ProcessInfo.processInfo.physicalMemory) / 1024 / 1024 // MB

        // 사용 가능한 메모리가 500MB 이상
        return (availableMemory - usedMemory) > 500
    }
}

extension UIDevice.BatteryState: CustomStringConvertible {
    public var description: String {
        switch self {
        case .unknown: return "Unknown"
        case .unplugged: return "Unplugged"
        case .charging: return "Charging"
        case .full: return "Full"
        @unknown default: return "Unknown"
        }
    }
}

extension ProcessInfo.ThermalState: CustomStringConvertible {
    public var description: String {
        switch self {
        case .nominal: return "Nominal"
        case .fair: return "Fair"
        case .serious: return "Serious"
        case .critical: return "Critical"
        @unknown default: return "Unknown"
        }
    }
}
```

## iOS 26 BGContinuedProcessingTask 구현

iOS 26+에서는 `BGContinuedProcessingTask`를 사용하여 사용자 경험을 크게 개선할 수 있습니다.

### BGContinuedProcessingTask 특징

- ✅ Foreground에서 시작하여 Background로 자연스럽게 전환
- ✅ Live Activity로 진행 상황 실시간 표시
- ✅ 사용자가 언제든 작업 취소 가능
- ✅ GPU 접근 가능 (Background GPU Access entitlement 필요)
- ✅ ProgressReporting 프로토콜로 진행률 보고

### 1. Info.plist 및 Entitlements 설정

**Info.plist:**
```xml
<key>BGTaskSchedulerPermittedIdentifiers</key>
<array>
    <string>com.yourapp.photoprocessing.continued</string>
</array>

<key>UIBackgroundModes</key>
<array>
    <string>processing</string>
</array>

<key>NSSupportsLiveActivities</key>
<true/>
```

**Entitlements (YourApp.entitlements):**
```xml
<key>com.apple.developer.background-processing.gpu-access</key>
<true/>
```

### 2. Live Activity 정의

```swift
import ActivityKit
import Foundation

// Live Activity Attributes
struct PhotoProcessingAttributes: ActivityAttributes {
    public struct ContentState: Codable, Hashable {
        var totalPhotos: Int
        var processedPhotos: Int
        var uploadedPhotos: Int
        var filteredPhotos: Int
        var currentPhase: ProcessingPhase
        var estimatedTimeRemaining: TimeInterval

        var progress: Double {
            guard totalPhotos > 0 else { return 0 }
            return Double(processedPhotos) / Double(totalPhotos)
        }
    }

    var startTime: Date
}

enum ProcessingPhase: String, Codable {
    case analyzing = "Analyzing Photos"
    case moderating = "Moderating Content"
    case uploading = "Uploading Photos"
    case completed = "Completed"
    case cancelled = "Cancelled"
}
```

### 3. Live Activity Widget

```swift
import SwiftUI
import WidgetKit
import ActivityKit

@available(iOS 26.0, *)
struct PhotoProcessingLiveActivity: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: PhotoProcessingAttributes.self) { context in
            // Lock Screen UI
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Image(systemName: "photo.stack")
                        .font(.title2)
                        .foregroundColor(.blue)

                    VStack(alignment: .leading, spacing: 4) {
                        Text("Photo Processing")
                            .font(.headline)

                        Text(context.state.currentPhase.rawValue)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Spacer()

                    Button(intent: CancelProcessingIntent()) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title2)
                            .foregroundColor(.red)
                    }
                }

                ProgressView(value: context.state.progress) {
                    HStack {
                        Text("\(context.state.processedPhotos)/\(context.state.totalPhotos)")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        Spacer()

                        Text("~\(formatTimeRemaining(context.state.estimatedTimeRemaining))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .tint(.blue)

                HStack(spacing: 16) {
                    StatLabel(icon: "checkmark.circle", value: context.state.uploadedPhotos, label: "Uploaded")
                    StatLabel(icon: "xmark.circle", value: context.state.filteredPhotos, label: "Filtered")
                }
            }
            .padding()
            .background(Color(.systemBackground))
            .activityBackgroundTint(Color.blue.opacity(0.1))
        } dynamicIsland: { context in
            // Dynamic Island UI
            DynamicIsland {
                DynamicIslandExpandedRegion(.leading) {
                    Image(systemName: "photo.stack")
                        .font(.title2)
                        .foregroundColor(.blue)
                }

                DynamicIslandExpandedRegion(.trailing) {
                    Text("\(Int(context.state.progress * 100))%")
                        .font(.title3)
                        .bold()
                }

                DynamicIslandExpandedRegion(.center) {
                    Text(context.state.currentPhase.rawValue)
                        .font(.caption)
                }

                DynamicIslandExpandedRegion(.bottom) {
                    ProgressView(value: context.state.progress)
                        .tint(.blue)
                }
            } compactLeading: {
                Image(systemName: "photo.stack")
                    .foregroundColor(.blue)
            } compactTrailing: {
                ProgressView(value: context.state.progress)
                    .tint(.blue)
                    .frame(width: 20)
            } minimal: {
                Image(systemName: "photo.stack")
                    .foregroundColor(.blue)
            }
        }
    }

    private func formatTimeRemaining(_ seconds: TimeInterval) -> String {
        let minutes = Int(seconds) / 60
        if minutes < 60 {
            return "\(minutes)m"
        } else {
            let hours = minutes / 60
            let remainingMinutes = minutes % 60
            return "\(hours)h \(remainingMinutes)m"
        }
    }
}

struct StatLabel: View {
    let icon: String
    let value: Int
    let label: String

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption)
            Text("\(value)")
                .font(.caption)
                .bold()
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }
}

// App Intent for cancellation
@available(iOS 26.0, *)
struct CancelProcessingIntent: AppIntent {
    static var title: LocalizedStringResource = "Cancel Processing"

    func perform() async throws -> some IntentResult {
        // Post notification to cancel processing
        NotificationCenter.default.post(
            name: NSNotification.Name("CancelPhotoProcessing"),
            object: nil
        )
        return .result()
    }
}
```

### 4. BGContinuedProcessingTask Service

```swift
import BackgroundTasks
import ActivityKit

@available(iOS 26.0, *)
class PhotoProcessingContinuedTask: NSObject {
    static let shared = PhotoProcessingContinuedTask()

    private var currentTask: BGContinuedProcessingTask?
    private var activity: Activity<PhotoProcessingAttributes>?
    private var isCancelled = false
    private let taskIdentifier = "com.yourapp.photoprocessing.continued"

    private override init() {
        super.init()
        setupCancellationObserver()
    }

    private func setupCancellationObserver() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleCancellation),
            name: NSNotification.Name("CancelPhotoProcessing"),
            object: nil
        )
    }

    @objc private func handleCancellation() {
        print("⚠️ User cancelled processing")
        isCancelled = true
        currentTask?.cancel()
    }

    func startProcessing(totalPhotos: Int) async throws {
        // Create Live Activity
        let initialState = PhotoProcessingAttributes.ContentState(
            totalPhotos: totalPhotos,
            processedPhotos: 0,
            uploadedPhotos: 0,
            filteredPhotos: 0,
            currentPhase: .analyzing,
            estimatedTimeRemaining: Double(totalPhotos) * 0.72 // ~0.72초/장 예상
        )

        let attributes = PhotoProcessingAttributes(startTime: Date())

        do {
            activity = try Activity<PhotoProcessingAttributes>.request(
                attributes: attributes,
                content: .init(state: initialState, staleDate: nil),
                pushType: nil
            )
            print("✅ Live Activity started")
        } catch {
            print("❌ Failed to start Live Activity: \(error)")
            throw error
        }

        // Create BGContinuedProcessingTask
        let progress = Progress(totalUnitCount: Int64(totalPhotos))

        currentTask = BGContinuedProcessingTask(
            identifier: taskIdentifier,
            using: progress
        )

        guard let task = currentTask else {
            throw ProcessingError.taskCreationFailed
        }

        // Start processing
        try await processPhotosWithProgress(task: task, totalPhotos: totalPhotos)
    }

    private func processPhotosWithProgress(
        task: BGContinuedProcessingTask,
        totalPhotos: Int
    ) async throws {
        let startTime = Date()
        var stats = ProcessingStats()
        stats.total = totalPhotos

        // Fetch photos
        await updateLiveActivity(
            stats: stats,
            phase: .analyzing,
            startTime: startTime
        )

        let photos = await fetchAllPhotos()

        // Process photos with GPU acceleration
        await updateLiveActivity(
            stats: stats,
            phase: .moderating,
            startTime: startTime
        )

        for (index, photo) in photos.enumerated() {
            guard !isCancelled else {
                print("⚠️ Processing cancelled by user")
                await updateLiveActivity(
                    stats: stats,
                    phase: .cancelled,
                    startTime: startTime
                )
                await endActivity()
                throw ProcessingError.cancelled
            }

            // Process single photo
            let result = await processSinglePhoto(photo)

            // Update stats
            stats.processed += 1
            switch result {
            case .uploaded:
                stats.uploaded += 1
            case .filtered:
                stats.filtered += 1
            case .failed:
                stats.failed += 1
            }

            // Update progress
            task.progress.completedUnitCount = Int64(index + 1)

            // Update Live Activity every 10 photos or on last photo
            if index % 10 == 0 || index == photos.count - 1 {
                await updateLiveActivity(
                    stats: stats,
                    phase: .moderating,
                    startTime: startTime
                )
            }
        }

        // Upload phase
        await updateLiveActivity(
            stats: stats,
            phase: .uploading,
            startTime: startTime
        )

        // Wait for uploads to complete
        try await waitForUploadsToComplete()

        // Completed
        await updateLiveActivity(
            stats: stats,
            phase: .completed,
            startTime: startTime
        )

        // Keep Live Activity visible for 5 seconds
        try await Task.sleep(nanoseconds: 5_000_000_000)

        await endActivity()

        print("""
        ✅ Processing completed successfully
           Total: \(stats.total)
           Processed: \(stats.processed)
           Uploaded: \(stats.uploaded)
           Filtered: \(stats.filtered)
           Failed: \(stats.failed)
        """)
    }

    private func updateLiveActivity(
        stats: ProcessingStats,
        phase: ProcessingPhase,
        startTime: Date
    ) async {
        guard let activity = activity else { return }

        let elapsed = Date().timeIntervalSince(startTime)
        let remaining = estimateTimeRemaining(stats: stats, elapsed: elapsed)

        let newState = PhotoProcessingAttributes.ContentState(
            totalPhotos: stats.total,
            processedPhotos: stats.processed,
            uploadedPhotos: stats.uploaded,
            filteredPhotos: stats.filtered,
            currentPhase: phase,
            estimatedTimeRemaining: remaining
        )

        await activity.update(
            ActivityContent(state: newState, staleDate: nil)
        )
    }

    private func estimateTimeRemaining(stats: ProcessingStats, elapsed: TimeInterval) -> TimeInterval {
        guard stats.processed > 0 else {
            return Double(stats.total) * 0.72
        }

        let timePerPhoto = elapsed / Double(stats.processed)
        let remaining = Double(stats.total - stats.processed) * timePerPhoto
        return remaining
    }

    private func endActivity() async {
        guard let activity = activity else { return }

        await activity.end(
            ActivityContent(
                state: activity.content.state,
                staleDate: Date()
            ),
            dismissalPolicy: .after(.now + 5)
        )

        self.activity = nil
    }

    private func fetchAllPhotos() async -> [PHAsset] {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let fetchOptions = PHFetchOptions()
                fetchOptions.sortDescriptors = [
                    NSSortDescriptor(key: "creationDate", ascending: false)
                ]

                let allPhotos = PHAsset.fetchAssets(with: .image, options: fetchOptions)
                var photos: [PHAsset] = []

                allPhotos.enumerateObjects { asset, _, _ in
                    photos.append(asset)
                }

                continuation.resume(returning: photos)
            }
        }
    }

    private func processSinglePhoto(_ photo: PHAsset) async -> PhotoProcessingResult {
        await withCheckedContinuation { continuation in
            PhotoProcessingService.shared.processPhoto(photo) { result in
                continuation.resume(returning: result)
            }
        }
    }

    private func waitForUploadsToComplete() async throws {
        // Wait for all pending uploads
        try await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
    }
}

enum ProcessingError: Error {
    case taskCreationFailed
    case cancelled
}
```

### 5. SwiftUI Integration for iOS 26

```swift
import SwiftUI

@available(iOS 26.0, *)
struct PhotoProcessingViewIOS26: View {
    @StateObject private var viewModel = PhotoProcessingViewModelIOS26()

    var body: some View {
        VStack(spacing: 20) {
            Text("Photo Processing (iOS 26+)")
                .font(.largeTitle)
                .bold()

            VStack(alignment: .leading, spacing: 12) {
                FeatureRow(
                    icon: "sparkles",
                    title: "GPU Acceleration",
                    description: "Fast processing with GPU"
                )

                FeatureRow(
                    icon: "antenna.radiowaves.left.and.right",
                    title: "Live Activity",
                    description: "Real-time progress on Lock Screen"
                )

                FeatureRow(
                    icon: "hand.tap",
                    title: "User Cancellable",
                    description: "Cancel anytime from Live Activity"
                )
            }
            .padding()
            .background(Color.blue.opacity(0.1))
            .cornerRadius(12)

            if viewModel.isProcessing {
                VStack(spacing: 12) {
                    ProgressView()
                        .scaleEffect(1.5)

                    Text("Processing in background...")
                        .font(.headline)

                    Text("Check Live Activity for progress")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
            }

            Button(action: {
                Task {
                    await viewModel.startProcessing()
                }
            }) {
                Text(viewModel.isProcessing ? "Processing..." : "Start Processing")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(viewModel.isProcessing ? Color.gray : Color.blue)
                    .cornerRadius(10)
            }
            .disabled(viewModel.isProcessing)
            .padding()

            if let error = viewModel.error {
                Text("Error: \(error)")
                    .font(.caption)
                    .foregroundColor(.red)
                    .padding()
            }

            Spacer()
        }
        .padding()
    }
}

struct FeatureRow: View {
    let icon: String
    let title: String
    let description: String

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(.blue)
                .frame(width: 30)

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline)
                    .bold()

                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }
}

@available(iOS 26.0, *)
class PhotoProcessingViewModelIOS26: ObservableObject {
    @Published var isProcessing = false
    @Published var error: String?

    func startProcessing() async {
        isProcessing = true
        error = nil

        do {
            // Request photo library access
            let status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
            guard status == .authorized else {
                error = "Photo library access denied"
                isProcessing = false
                return
            }

            // Get total photo count
            let fetchOptions = PHFetchOptions()
            let totalPhotos = PHAsset.fetchAssets(with: .image, options: fetchOptions).count

            // Start BGContinuedProcessingTask with Live Activity
            try await PhotoProcessingContinuedTask.shared.startProcessing(
                totalPhotos: totalPhotos
            )

            isProcessing = false
        } catch {
            self.error = error.localizedDescription
            isProcessing = false
            print("❌ Processing failed: \(error)")
        }
    }
}
```

### 6. AppDelegate 설정

```swift
import UIKit
import BackgroundTasks

@main
class AppDelegate: UIResponder, UIApplicationDelegate {

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        // Register BGContinuedProcessingTask for iOS 26+
        if #available(iOS 26.0, *) {
            BGTaskScheduler.shared.register(
                forTaskWithIdentifier: "com.yourapp.photoprocessing.continued",
                using: nil
            ) { task in
                print("⚠️ BGContinuedProcessingTask launched by system")
                // This task is user-initiated, so typically started from app
                // System may call this if app was terminated while processing
            }
        }

        return true
    }
}
```

### 사용 방법

**iOS 26+ 사용자:**

1. 앱에서 "Start Processing" 버튼 클릭
2. Live Activity가 Lock Screen과 Dynamic Island에 표시됨
3. 앱을 백그라운드로 전환해도 처리 계속
4. Live Activity에서 실시간 진행 상황 확인
5. 필요시 Live Activity에서 취소 버튼으로 중단

**장점:**
- ✅ 사용자가 진행 상황을 항상 확인 가능
- ✅ GPU 가속으로 빠른 처리 (30-60분)
- ✅ 언제든 취소 가능
- ✅ 자연스러운 사용자 경험

## 성능 비교표

| 항목 | iOS 18 | iOS 26 (BGProcessingTask) | iOS 26 (BGContinuedProcessingTask) |
|------|--------|---------------------------|-----------------------------------|
| **GPU 백그라운드 사용** | ❌ 제한적 | ✅ 지원 | ✅ 지원 |
| **Live Activity** | ❌ | ❌ | ✅ |
| **사용자 취소** | ❌ | ❌ | ✅ |
| **시작 방법** | 시스템 스케줄링 | 시스템 스케줄링 | 사용자가 Foreground에서 시작 |
| **Compute Unit** | `.cpuOnly` | `.all` | `.all` |
| **동시 처리 수** | 3-5개 | 10-20개 | 10-20개 |
| **배치 크기** | 10장 | 50장 | 50장 |
| **예상 처리 시간** (5000장) | 2-3시간 | 30-60분 | 30-60분 |
| **진행 상황 표시** | ❌ | ❌ | ✅ Real-time |
| **배터리 영향** | 중간 | 높음 | 높음 |
| **Metal 지원** | ❌ | ✅ | ✅ |
| **MPS 사용** | ❌ | ✅ | ✅ |
| **추천 사용 사례** | 자동 동기화 | 야간 배치 작업 | 사용자 주도 긴 작업 |
| **추천 실행 조건** | 충전 중 + WiFi | 충전 중 + WiFi | 사용자 시작 시 |

## 최종 권장사항

### iOS 18 (BGProcessingTask)
- **사용 사례**: 자동 동기화, 야간 배치 작업
- **최적화 전략**:
  - CPU 최적화에 집중
  - 작은 배치 크기 사용 (10-20장)
  - 동시 처리 제한 (3-5개)
  - 긴 처리 시간 예상 (2-3시간)
  - 사용자에게 충분한 시간 제공
- **실행 조건**: 충전 중 + WiFi + 시스템이 최적 시간 선택

### iOS 26 (BGProcessingTask)
- **사용 사례**: 자동 동기화, 야간 배치 작업
- **최적화 전략**:
  - GPU 가속 활용
  - 큰 배치 크기 사용 (50-100장)
  - 많은 동시 처리 (10-20개)
  - 빠른 처리 시간 (30-60분)
  - Metal/MPS로 전처리 최적화
- **실행 조건**: 충전 중 + WiFi + 시스템이 최적 시간 선택

### iOS 26 (BGContinuedProcessingTask) ⭐ 추천
- **사용 사례**: **사용자가 시작하는 긴 작업** (사진 처리, 대용량 파일 변환 등)
- **주요 장점**:
  - ✅ Live Activity로 실시간 진행 상황 표시
  - ✅ Dynamic Island 지원
  - ✅ 사용자가 언제든 취소 가능
  - ✅ Foreground → Background 자연스러운 전환
  - ✅ GPU 가속 지원
  - ✅ 투명한 사용자 경험
- **최적화 전략**:
  - GPU 가속 활용 (`.all` + `allowsBackgroundGPUCompute`)
  - 큰 배치 크기 (50-100장)
  - 많은 동시 처리 (10-20개)
  - Progress 객체로 진행률 정확히 보고
  - Live Activity UI 최적화
  - 사용자 취소 처리 구현 필수
- **실행 조건**: 사용자가 앱에서 직접 시작
- **권장 시나리오**:
  - ✅ 5,000장 사진 moderation & upload
  - ✅ 대용량 비디오 편집/변환
  - ✅ 대량 데이터 암호화/복호화
  - ✅ 긴 ML 모델 학습/추론 작업

### 앱별 추천 방식

**사진 Moderation & Upload 앱 (5,000장):**
- **iOS 18**: `BGProcessingTask` (2-3시간 소요)
- **iOS 26**: `BGContinuedProcessingTask` ⭐ (30-60분 소요, Live Activity 지원)

**뉴스/소셜 앱 (자동 컨텐츠 동기화):**
- **iOS 18/26**: `BGAppRefreshTask` (30초, 가벼운 작업)

**파일 다운로드/업로드 앱:**
- **iOS 18/26**: `URLSession Background Transfer` (앱 종료되어도 계속)

**음악/팟캐스트 앱:**
- **iOS 18/26**: `Background Modes - Audio` (지속적 실행)

**피트니스 트래킹 앱:**
- **iOS 18/26**: `Background Modes - Location` (지속적 위치 추적)
