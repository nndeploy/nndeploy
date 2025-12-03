import nndeploy._nndeploy_internal as _C
OCRResult = _C.ocr.OCRResult
DrawDetectorBox = _C.ocr.DrawDetectorBox
RotateCropImage = _C.ocr.RotateCropImage
RotateImage180 = _C.ocr.RotateImage180
try:
    DetectorPreProcessParam = _C.ocr.DetectorPreProcessParam
    DetectorPreProcess = _C.ocr.DetectorPreProcess
    DetectorPostParam = _C.ocr.DetectorPostParam
    DetectorPostProcess = _C.ocr.DetectorPostProcess
    DetectorGraph = _C.ocr.DetectorGraph
    
    ClassifierPreProcessParam = _C.ocr.ClassifierPreProcessParam
    ClassifierPreProcess = _C.ocr.ClassifierPreProcess
    ClassifierPostParam = _C.ocr.ClassifierPostParam
    ClassifierPostProcess = _C.ocr.ClassifierPostProcess
    ClassifierGraph = _C.ocr.ClassifierGraph

    RecognizerPreProcessParam = _C.ocr.RecognizerPreProcessParam
    RecognizerPreProcess = _C.ocr.RecognizerPreProcess
    RecognizerPostParam = _C.ocr.RecognizerPostParam
    RecognizerPostProcess = _C.ocr.RecognizerPostProcess
    RecognizerGraph = _C.ocr.RecognizerGraph

    
    
    OcrText = _C.ocr.OcrText
    PrintOcrNode = _C.ocr.PrintOcrNode
    PrintOcrNodeParam = _C.ocr.PrintOcrNodeParam
except:
    pass