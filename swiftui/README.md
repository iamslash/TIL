- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [Swift Event](#swift-event)
  - [`@State`](#state)
  - [`@Published`](#published)
  - [`@State` vs `@Published`](#state-vs-published)
  - [`@Environment`](#environment)
- [Typical `@Environment`](#typical-environment)
- [`@StateObject` vs `@ObservedObject`](#stateobject-vs-observedobject)
  - [`@Binding`](#binding)

---

# Abstract

Interface Builder 다음의 UI Tool 이다.

# Materials

* [SwiftUI Bootcamp (Beginner Level) | youtube](https://www.youtube.com/playlist?list=PLwvDm4VfkdphqETTBf-DdjCoAvhai1QpO)
  * [src](https://github.com/SwiftfulThinking/SwiftUI-Bootcamp)
* [SwiftUI | Apple](https://developer.apple.com/tutorials/swiftui/creating-and-combining-views)
  * [src](https://docs-assets.developer.apple.com/published/9637262be4dfa3661d596e567d0c793f/CreatingAndCombiningViews.zip)
* [SwiftUI Masterclass 2023 - iOS App Development & Swift | udemy](https://www.udemy.com/course/swiftui-masterclass-course-ios-development-with-swift/)

# Basic

## Swift Event

* [Send Events from SwiftUI to UIKit and Vice Versa](https://www.swiftjectivec.com/events-from-swiftui-to-uikit-and-vice-versa/)

## `@State`

`@State` is a property wrapper in SwiftUI, used to manage the state of an iOS app's user interface. It allows you to create a mutable source of truth for data that can be bound and observed by the SwiftUI view. When the data stored in a `@State` property is modified, SwiftUI automatically updates the view to reflect the new state of the data.

You use `@State` with simple value types like `Bool`, `String`, and `Int` that are owned by a single view and should be updated instantly when changed. It is generally used for managing internal, private state of a view.

```swift
import SwiftUI

struct ContentView: View {
    @State private var isSwitchOn: Bool = false

    var body: some View {
        Toggle("Switch", isOn: $isSwitchOn)
    }
}
```

The `$` sign is used to create a two-way binding between the property and the Toggle view, ensuring that changes in the state of the switch are automatically reflected in the view.

## `@Published`

`@Published` is a property wrapper in SwiftUI and part of the Combine framework, used in conjunction with ObservableObject to handle data changes and updates in an app's state. When you place the `@Published` property wrapper before a property in a class that conforms to the `ObservableObject` protocol, it automatically announces any changes made to that property by sending notifications to any subscribers.

This enables SwiftUI to efficiently re-render your views whenever the bound data changes, allowing you to separate your app state and logic from the view layer.

```swift
import SwiftUI
import Combine

class Counter: ObservableObject {
    @Published var count: Int = 0
}

struct ContentView: View {
    @ObservedObject private var counter = Counter()

    var body: some View {
        VStack {
            Text("Count: \(counter.count)")
            Button("Increment") {
                counter.count += 1
            }
        }
    }
}
```

In this example, the `Counter` class conforms to the `ObservableObject` protocol, and `count` property is marked as `@Published`. This means that whenever count property changes, it will notify its subscribers. In the ContentView, we are using `@ObservedObject` to create a binding between the view and the counter object. When the count is increased by tapping the "Increment" button, the view automatically updates to reflect the new value of the count property.

## `@State` vs `@Published`

Both `@State` and `@Published` are used to manage and observe data changes in
SwiftUI, but they serve different purposes and have different use cases.

`@State`:
1. It is a property wrapper for value types like `Bool`, `String`, and `Int`.
2. It is used for managing the private, internal state of a single view.
3. When a `@State` property is changed, the view is automatically re-rendered.
4. Ideal for simple data used within a single view or small components where the
   data does not need to be shared across multiple views.

`@Published`:
1. It is a property wrapper for properties in classes that conform to
   `ObservableObject`.
2. It is designed to manage the data that needs to be shared across multiple
   views or components.
3. Using `@Published` with `ObservableObject` allows views to subscribe and get
   notified when the property is changed, which leads to updating the views.
4. Ideal for larger or more complex apps where data and state need to be
   separated from the view and shared between different views or app components.

In summary, use `@State` when you need a simple data storage for managing the
internal state of a **single view**, and use `@Published` when you need to manage
and share data across **multiple views** using a separate data model conforming to
`ObservableObject`.

## `@Environment`

`@Environment` is a property wrapper in SwiftUI that allows you to access and use
environment values (shared data and settings) that are managed by the SwiftUI
system. These environment values can represent various aspects of the
environment, such as system settings, user preferences, or context-specific
information provided by parent views.

Using `@Environment` allows you to decouple the configuration of your views from
their content, which helps create more reusable views. Some typical environment
values include `colorScheme`, `locale`, `managedObjectContext`, and `presentationMode`.
Parent views can also create and provide custom environment values for their
child views.

```swift
import SwiftUI

struct ContentView: View {
    @Environment(\.colorScheme) var colorScheme

    var body: some View {
        VStack {
            if colorScheme == .dark {
                Text("Dark Mode is On")
            } else {
                Text("Light Mode is On")
            }
        }
    }
}
```

In this example, the `@Environment` property wrapper is used to get the current
color scheme (dark or light mode) applied on the device. Based on the current
color scheme, the corresponding text is displayed. When the color scheme
changes, SwiftUI automatically updates the view to reflect the new state.


# Typical `@Environment`

There are several predefined `@Environment` values in SwiftUI that allow access
to system settings, user preferences, or context-specific information. Here's a
list of some common `@Environment` properties:

1. **.colorScheme**: Represents the current color scheme (light or dark mode) applied on the device.
2. **.locale**: Represents the current locale settings, which affect language,
   date formatting, and other regional settings.
3. **.layoutDirection**: Represents the layout direction (left-to-right or right-to-left) based on the device or app configuration.
4. **.sizeCategory**: Represents the current dynamic type (text size) setting on
   the device.
5. **.calendar**: Represents the current calendar used for date calculations and
   formatting.
6. **.managedObjectContext**: Represents the Core Data managed object context
   when using Core Data in your app.
7. **.presentationMode**: Represents the current presentation mode of a sheet or
   a navigation view. You can use it to programmatically dismiss a presented
   view.
8. **.horizontalSizeClass** and **.verticalSizeClass**: Represent user interface
   size classes used for adapting the layout to different screen sizes and
   orientations.

These are just a few examples of the predefined environment values SwiftUI
provides. To use a specific environment value, you can access it using the
`@Environment` property wrapper followed by the key path of the desired value.
For example, to access the color scheme, you would use
`@Environment(\.colorScheme) var colorScheme`

# `@StateObject` vs `@ObservedObject`

`@StateObject` and `@ObservedObject` are both property wrappers in SwiftUI that
help you manage and observe state changes in your app. However, they serve
different purposes and have specific use cases.

`@StateObject`:
1. Used with a class that conforms to the `ObservableObject` protocol.
2. Creates and owns the instance of the observable object.
3. Responsible for managing the lifecycle of the object and maintaining a single
   source of truth, ensuring the object is only created once.
4. Suitable for use in a parent view where you want to initialize and manage the
   object's life cycle.
5. When a property marked with `@Published` in a `@StateObject` changes, SwiftUI
   updates the views using this object.

`@ObservedObject`:
1. Used with a class that conforms to the `ObservableObject` protocol.
2. Does not manage or own the instance of the observable object, only observes
   it.
3. Assumes that the object is created and owned externally, typically by a
   parent view or an app's global state.
4. Suitable for use in child views to access and update the observable object
   passed down from a parent or ancestor view.
5. When a property marked with `@Published` in a `@ObservedObject` changes,
   SwiftUI updates the views using this object.

In summary, use `@StateObject` when you need to initialize and manage the
lifecycle of an observable object in a view, while use `@ObservedObject` when
you need to observe and update an object that is created and owned externally,
such as by a parent or ancestor view.

## `@Binding`

`@Binding` is a property wrapper in SwiftUI used to create a mutable two-way
binding between a property and a view. It enables a child view to access, share,
and modify a value owned by a parent or ancestor view without directly owning
the underlying state of the property.

When you use `@Binding` with a property in a child view, it reflects the changes
made in that view into the original data source in the parent or ancestor view,
allowing you to create more decoupled and reusable components in SwiftUI.

Here's an example to demonstrate the usage of `@Binding`:

```swift
import SwiftUI

struct ContentView: View {
    @State private var isSwitchOn: Bool = false

    var body: some View {
        VStack {
            Toggle("Switch", isOn: $isSwitchOn)
            ChildView(isSwitchOn: $isSwitchOn)
        }
    }
}

struct ChildView: View {
    @Binding var isSwitchOn: Bool

    var body: some View {
        Button("Toggle Switch") {
            isSwitchOn.toggle()
        }
    }
}
```

the ContentView has a `@State` property `isSwitchOn`. The ChildView uses an `@Binding`
property with the same name to create a two-way binding with the parent view's
property. When the button in the `ChildView` is tapped, it toggles the value of
`isSwitchOn`, which is also reflected in the parent view's property due to the
binding.
