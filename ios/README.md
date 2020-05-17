# TensorFlow Lite Object Detection iOS App with YOLOv3-tiny

**iOS Versions Supported:** iOS 12.0 and above.
**Xcode Version Required:** 10.0 and above

This application is modified from [Tensorflow's example application](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/ios).

## Overview

This is a camera app that continuously detects the objects (bounding boxes and classes) in the frames seen by your device's back camera, using our quantized [YOLOv3-tiny](https://github.com/Morris88826/comp4471/tree/master/yolov3) model trained on the [COCO dataset](http://cocodataset.org/). These instructions walk you through building and running the demo on an iOS device.

You have to follow the [YOLOv3-tiny](https://github.com/Morris88826/comp4471/tree/master/yolov3) script to get `yolov3-tiny.tflite` and `coco.names` and put them in `comp4471/ios/ObjectDetection/Model/` first, and then follow the instructions below.

## Prerequisites

* You must have Xcode installed

* You must have a valid Apple ID, but it does not need to be a *Developer* (i.e. paid) one if you just want to dev/test the app on your device. <!--Is Mac necessary though?-->

* The demo app requires a camera and must be executed on a real iOS device. You can build it and run with the iPhone Simulator but the app raises a camera not found exception. (In short: Go get an iPhone/iPad.)

* You don't need to build the entire TensorFlow library to run the demo, it uses CocoaPods to download the TensorFlow Lite library.

* You'll also need the Xcode command-line tools:
 ```xcode-select --install```
 If this is a new install, you will need to run the Xcode application once to agree to the license before continuing.
 
## Building the iOS Demo App

1. Install CocoaPods if you don't have it.
```sudo gem install cocoapods```

2. Install the pod to generate the workspace file:
```cd lite/examples/object_detection/ios/```
```pod install```
  If you have installed this pod before and that command doesn't work, try
```pod update```
At the end of this step you should have a file called ```ObjectDetection.xcworkspace```

3. Open **ObjectDetection.xcworkspace** in Xcode.

4. Please change the bundle identifier to a unique identifier and select your development team in **'General->Signing'** before building the application if you are using an iOS device. 
   * Click the folder icon on the top-left corner ("Show the project navigator")
   * Click the first `ObjectDetection` next to the blue xcode icon
   * Click the tab "Signing & Capabilities"
   * Change the `Team` to yourself
   * Change the `Bundle Identifier` to a unique one. (Friendly reminder: you can create only 10 IDs every 7 days, so you may not want to waste too many of them.)

5. Build and run the app in Xcode.
You'll have to grant permissions for the app to use the device's camera. Point the camera at various objects and enjoy seeing how the model classifies things!
   * Go to `Settings > General > Device Management` and verify our app.

## Model Used

This app uses the [YOLOv3-tiny](https://github.com/Morris88826/comp4471/tree/master/yolov3) model trained on [COCO dataset](http://cocodataset.org/). The input image size required is 416 X 416 X 3. 

## iOS App Details

The app is written entirely in Swift and uses the TensorFlow Lite
[Swift library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/swift).
