def extractPrediction(prediction):
    my_dict = {}
    my_dict['label'] = prediction['label']
    width = prediction['bottomright']['x'] - prediction['topleft']['x']
    height = prediction['bottomright']['y'] - prediction['topleft']['y']
    my_dict['ROI'] = {'origin': prediction['topleft'], "width":width, "height":height}
    return my_dict

def findCommonItems(left_img, right_img):
    info_l = []
    info_r = []
    info = []

    for l in left_img:
        info_l.append(extractPrediction(l))

    for r in right_img:
        info_r.append(extractPrediction(r))


    for l in info_l:
        for r in info_r:
            if l['label'] == r['label']:
                m_dict = {}
                m_dict['label'] = l['label']
                m_dict['left_ROI'] = l['ROI']
                m_dict['right_ROI'] = r['ROI']
                info.append(m_dict)
    
    return info