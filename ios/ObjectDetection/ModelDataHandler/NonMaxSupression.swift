//
//  NonMaxSupression.Swift
//  ObjectDetection
//
//  Created by Shang-Ling Hsu on 2020/5/17.
//  Copyright © 2020 Y Media Labs. All rights reserved.
//

/*
  Copyright (c) 2017-2019 M.I. Hollemans

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to
  deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
*/

import Foundation
import Accelerate

public struct BoundingBox {
  /** Index of the predicted class. */
  public let classIndex: Int

  /** Confidence score. */
  public let confidence: Float

  /** Coordinates between 0 and resolution-1. */
  public let rect: CGRect

  public init(classIndex: Int, confidence: Float, rect: CGRect) {
    self.classIndex = classIndex
    self.confidence = confidence
    self.rect = rect
  }
}

/**
  Computes intersection-over-union overlap between two bounding boxes.
*/
public func IOU(_ a: CGRect, _ b: CGRect) -> Float {
  let areaA = a.width * a.height
  if areaA <= 0 { return 0 }

  let areaB = b.width * b.height
  if areaB <= 0 { return 0 }

  let intersectionMinX = max(a.minX, b.minX)
  let intersectionMinY = max(a.minY, b.minY)
  let intersectionMaxX = min(a.maxX, b.maxX)
  let intersectionMaxY = min(a.maxY, b.maxY)
  let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
                         max(intersectionMaxX - intersectionMinX, 0)
//  return Float(intersectionArea / (areaA + areaB - intersectionArea))
  return Float(intersectionArea / min(areaA, areaB))
}

// No classes difference
public func nonMaxSuppression(boundingBoxes: [BoundingBox],
                              num_classes: Int,
                              confidence: Float,
                              nms_threshold: Float) -> [BoundingBox] {
  let confidentBoxes: [BoundingBox] = boundingBoxes.filter {$0.confidence > confidence}

  var selected: [BoundingBox] = []
    
  for c in 0...num_classes-1 {
    var classSelected: [BoundingBox] = []
    var classBoxes: [BoundingBox] = confidentBoxes.filter { $0.classIndex == c }
    
    // Sort the boxes based on their confidence scores, from high to low.
    classBoxes.sort {$0.confidence > $1.confidence}

    // Loop through the bounding boxes, from highest score to lowest score,
    // and determine whether or not to keep each box.
    for boxA in classBoxes {

      var shouldSelect = true

      // Does the current box overlap one of the selected boxes more than the
      // given threshold amount? Then it's too similar, so don't keep it.
      for boxB in classSelected {
        if IOU(boxA.rect, boxB.rect) > nms_threshold {
          shouldSelect = false
          break
        }
      }

      // This bounding box did not overlap too much with any previously selected
      // bounding box, so we'll keep it.
      if shouldSelect {
        selected.append(boxA)
        classSelected.append(boxA)
      }
    }
  }
  
  return selected
}
