import UIKit

/**
 This UIView draws overlay on a detected object.
 */
class DepthView: UIView {

  var map: Data = Data()
  private let myAlpha: CGFloat = 0.7

  override func draw(_ rect: CGRect){
    drawBackground(of: map)
  }

  /**
   This method draws the background of the string.
   */
  func drawBackground(of map: Data) {

    let rect = CGRect(x: 0, y: 0 , width: 1080, height: 1920)

    let stringBgPath = UIBezierPath(rect: rect)
//    picture.withAlphaComponent(myAlpha).setFill()
    stringBgPath.fill()
  }

}
