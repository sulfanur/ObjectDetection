package com.example.yolov5tfliteandroid.analysis

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.widget.ImageView
import android.widget.TextView
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import com.example.yolov5tfliteandroid.detector.Yolov5TFLiteDetector
import com.example.yolov5tfliteandroid.utils.ImageProcess
import io.reactivex.rxjava3.android.schedulers.AndroidSchedulers
import io.reactivex.rxjava3.core.Observable
import io.reactivex.rxjava3.core.ObservableEmitter
import io.reactivex.rxjava3.schedulers.Schedulers

class FullImageAnalyse(
    context: Context?,
    var previewView: PreviewView,
    var boxLabelCanvas: ImageView,
    var rotation: Int,
    private val inferenceTimeTextView: TextView,
    private val frameSizeTextView: TextView,
    yolov5TFLiteDetector: Yolov5TFLiteDetector
) : ImageAnalysis.Analyzer {
    class Result(var costTime: Long, var bitmap: Bitmap)

    var imageProcess: ImageProcess
    private val yolov5TFLiteDetector: Yolov5TFLiteDetector

    init {
        imageProcess = ImageProcess()
        this.yolov5TFLiteDetector = yolov5TFLiteDetector
    }

    override fun analyze(image: ImageProxy) {
        val previewHeight = previewView.height
        val previewWidth = previewView.width

        // 这里Observable将image analyse的逻辑放到子线程计算, 渲染UI的时候再拿回来对应的数据, 避免前端UI卡顿
        Observable.create { emitter: ObservableEmitter<Result> ->
            val start = System.currentTimeMillis()
            val yuvBytes = arrayOfNulls<ByteArray>(3)
            val planes = image.planes
            val imageHeight = image.height
            val imagewWidth = image.width
            imageProcess.fillBytes(planes, yuvBytes)
            val yRowStride = planes[0].rowStride
            val uvRowStride = planes[1].rowStride
            val uvPixelStride = planes[1].pixelStride
            val rgbBytes = IntArray(imageHeight * imagewWidth)
            imageProcess.YUV420ToARGB8888(
                yuvBytes[0]!!,
                yuvBytes[1]!!,
                yuvBytes[2]!!,
                imagewWidth,
                imageHeight,
                yRowStride,
                uvRowStride,
                uvPixelStride,
                rgbBytes
            )

            // 原图bitmap
            val imageBitmap = Bitmap.createBitmap(imagewWidth, imageHeight, Bitmap.Config.ARGB_8888)
            imageBitmap.setPixels(rgbBytes, 0, imagewWidth, 0, 0, imagewWidth, imageHeight)

            // 图片适应屏幕fill_start格式的bitmap
            val scale = Math.max(
                previewHeight / (if (rotation % 180 == 0) imagewWidth else imageHeight).toDouble(),
                previewWidth / (if (rotation % 180 == 0) imageHeight else imagewWidth).toDouble()
            )
            val fullScreenTransform = imageProcess.getTransformationMatrix(
                imagewWidth,
                imageHeight,
                (scale * imageHeight).toInt(),
                (scale * imagewWidth).toInt(),
                if (rotation % 180 == 0) 90 else 0,
                false
            )

            // 适应preview的全尺寸bitmap
            val fullImageBitmap = Bitmap.createBitmap(
                imageBitmap,
                0,
                0,
                imagewWidth,
                imageHeight,
                fullScreenTransform,
                false
            )
            // 裁剪出跟preview在屏幕上一样大小的bitmap
            val cropImageBitmap =
                Bitmap.createBitmap(fullImageBitmap, 0, 0, previewWidth, previewHeight)

            // 模型输入的bitmap
            val previewToModelTransform = imageProcess.getTransformationMatrix(
                cropImageBitmap.width, cropImageBitmap.height,
                yolov5TFLiteDetector.inputSize.width,
                yolov5TFLiteDetector.inputSize.height,
                0, false
            )
            val modelInputBitmap = Bitmap.createBitmap(
                cropImageBitmap, 0, 0,
                cropImageBitmap.width, cropImageBitmap.height,
                previewToModelTransform, false
            )
            val modelToPreviewTransform = Matrix()
            previewToModelTransform.invert(modelToPreviewTransform)
            val recognitions = yolov5TFLiteDetector.detect(modelInputBitmap)
            //            ArrayList<Recognition> recognitions = yolov5TFLiteDetector.detect(imageBitmap);
            val emptyCropSizeBitmap =
                Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888)
            val cropCanvas = Canvas(emptyCropSizeBitmap)
            //            Paint white = new Paint();
//            white.setColor(Color.WHITE);
//            white.setStyle(Paint.Style.FILL);
//            cropCanvas.drawRect(new RectF(0,0,previewWidth, previewHeight), white);
            // 边框画笔
            val boxPaint = Paint()
            boxPaint.strokeWidth = 5f
            boxPaint.style = Paint.Style.STROKE
            boxPaint.color = Color.RED
            // 字体画笔
            val textPain = Paint()
            textPain.textSize = 50f
            textPain.color = Color.RED
            textPain.style = Paint.Style.FILL
            for (res in recognitions) {
                val location = res!!.getLocation()
                val label = res.labelName
                val confidence = res.confidence!!
                modelToPreviewTransform.mapRect(location)
                cropCanvas.drawRect(location, boxPaint)
                cropCanvas.drawText(
                    label + ":" + String.format("%.2f", confidence),
                    location.left,
                    location.top,
                    textPain
                )
            }
            val end = System.currentTimeMillis()
            val costTime = end - start
            image.close()
            emitter.onNext(Result(costTime, emptyCropSizeBitmap))
        }.subscribeOn(Schedulers.io()) // 这里定义被观察者,也就是上面代码的线程, 如果没定义就是主线程同步, 非异步
            // 这里就是回到主线程, 观察者接受到emitter发送的数据进行处理
            .observeOn(AndroidSchedulers.mainThread()) // 这里就是回到主线程处理子线程的回调数据.
            .subscribe { result: Result ->
                boxLabelCanvas.setImageBitmap(result.bitmap)
                frameSizeTextView.text = previewHeight.toString() + "x" + previewWidth
                inferenceTimeTextView.text = java.lang.Long.toString(result.costTime) + "ms"
            }
    }
}