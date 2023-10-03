package com.example.yolov5tfliteandroid.detector

import android.app.Activity
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Build
import android.util.Log
import android.util.Size
import android.widget.Toast
import com.example.yolov5tfliteandroid.utils.Recognition
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.DequantizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.common.ops.QuantizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.metadata.MetadataExtractor
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.util.Arrays
import java.util.PriorityQueue

class Yolov5TFLiteDetector {
    val inputSize = Size(640, 640)
    val outputSize = intArrayOf(1, 25200, 9)
    private val DETECT_THRESHOLD = 0.3f
    private val IOU_THRESHOLD = 0.25f
    private val IOU_CLASS_DUPLICATED_THRESHOLD = 0.7f
    private var tflite: Interpreter? = null
    private var model = "test_best_cpu.tflite"
    private var label = "label.txt"
    private var associatedAxisLabels: List<String>? = null
    var options = Interpreter.Options()

    fun initialModel(activity: Context?) {

        try {
            val tfliteModel: ByteBuffer = FileUtil.loadMappedFile(activity!!, model)
            tflite = Interpreter(tfliteModel, options)
            Log.i("tfliteSupport", "Success reading model: $model")

            associatedAxisLabels = FileUtil.loadLabels(activity, label)
            Log.i("tfliteSupport", "Success reading label: " + label)

        } catch (e: IOException) {
            Log.e("tfliteSupport", "Error reading model or label: ", e)
            Toast.makeText(activity, "load model error: " + e.message, Toast.LENGTH_LONG).show()
        }
    }

    fun detect(bitmap: Bitmap?): ArrayList<Recognition> {

        val imageProcessor: ImageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputSize.height, inputSize.width, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 200f))
            .build()
        var yolov5sTfliteInput: TensorImage = TensorImage(DataType.FLOAT32)

        yolov5sTfliteInput.load(bitmap)
        yolov5sTfliteInput = imageProcessor.process(yolov5sTfliteInput)

        var probabilityBuffer: TensorBuffer = TensorBuffer.createFixedSize(outputSize, DataType.FLOAT32)

        if (null != tflite) {
            // Здесь tflite по умолчанию добавит широту batch=1
            tflite!!.run(yolov5sTfliteInput.buffer, probabilityBuffer.buffer)
        }

        val recognitionArray = probabilityBuffer.floatArray
        val allRecognitions = ArrayList<Recognition>()
        for (i in 0 until outputSize[1]) {
            val gridStride = i * outputSize[2]
            // Поскольку при экспорте tflite автор yolov5 разделил вывод на размер изображения, здесь нужно умножить его обратно
            val x = recognitionArray[0 + gridStride] * inputSize.width
            val y = recognitionArray[1 + gridStride] * inputSize.height
            val w = recognitionArray[2 + gridStride] * inputSize.width
            val h = recognitionArray[3 + gridStride] * inputSize.height
            val xmin = Math.max(0.0, x - w / 2.0).toInt()
            val ymin = Math.max(0.0, y - h / 2.0).toInt()
            val xmax = Math.min(inputSize.width.toDouble(), x + w / 2.0).toInt()
            val ymax = Math.min(inputSize.height.toDouble(), y + h / 2.0).toInt()
            val confidence = recognitionArray[4 + gridStride]
            val classScores = Arrays.copyOfRange(recognitionArray, 5 + gridStride, 85 + gridStride)

            var labelId = 0
            var maxLabelScores = 0f
            for (j in classScores.indices) {
                if (classScores[j] > maxLabelScores) {
                    maxLabelScores = classScores[j]
                    labelId = j
                }
            }
            val r = Recognition(
                labelId,
                "",
                maxLabelScores,
                confidence,
                RectF(xmin.toFloat(), ymin.toFloat(), xmax.toFloat(), ymax.toFloat())
            )
            allRecognitions.add(
                r
            )
        }

        val nmsRecognitions = nms(allRecognitions)

        val nmsFilterBoxDuplicationRecognitions = nmsAllClass(nmsRecognitions)

        for (recognition in nmsFilterBoxDuplicationRecognitions) {
            val labelId = recognition.labelId
            val labelName = associatedAxisLabels!![labelId]
            recognition.labelName = labelName
        }
        return nmsFilterBoxDuplicationRecognitions
        Log.i("tfliteSupport", "recognize nms data size: "+nmsFilterBoxDuplicationRecognitions.toString())
    }

    /**
     * Неэкстремальное ингибирование
     *
     * @param allRecognitions
     * @return
     */
    protected fun nms(allRecognitions: ArrayList<Recognition>): ArrayList<Recognition> {
        val nmsRecognitions = ArrayList<Recognition>()

        for (i in 0 until outputSize[2] - 5) {
            val pq = PriorityQueue<Recognition>(
                25200
            ) { l, r -> // Intentionally reversed to put high confidence at the head of the queue.
                java.lang.Float.compare(r.confidence!!, l.confidence!!)
            }

            for (j in allRecognitions.indices) {
//                if (allRecognitions.get(j).getLabelId() == i) {
                if (allRecognitions[j].labelId == i && allRecognitions[j].confidence!! > DETECT_THRESHOLD) {
                    pq.add(allRecognitions[j])
                    //                    Log.i("tfliteSupport", allRecognitions.get(j).toString());
                }
            }

            while (pq.size > 0) {
                val a = arrayOfNulls<Recognition>(pq.size)
                val detections: Array<Recognition> = pq.toArray(a)
                val max = detections[0]
                nmsRecognitions.add(max)
                pq.clear()
                for (k in 1 until detections.size) {
                    val detection = detections[k]
                    if (boxIou(max.getLocation(), detection.getLocation()) < IOU_THRESHOLD) {
                        pq.add(detection)
                    }
                }
            }
        }
        return nmsRecognitions
    }

    /**
     * Неэкстремальное подавление всех данных независимо от категории
     *
     * @param allRecognitions
     * @return
     */
    protected fun nmsAllClass(allRecognitions: ArrayList<Recognition>): ArrayList<Recognition> {
        val nmsRecognitions = ArrayList<Recognition>()
        val pq = PriorityQueue<Recognition>(
            4
        ) { l, r -> // Intentionally reversed to put high confidence at the head of the queue.
            java.lang.Float.compare(r.confidence!!, l.confidence!!)
        }

        for (j in allRecognitions.indices) {
            if (allRecognitions[j].confidence!! > DETECT_THRESHOLD) {
                pq.add(allRecognitions[j])
            }
        }
        while (pq.size > 0) {
            val a = arrayOfNulls<Recognition>(pq.size)
            val detections: Array<Recognition> = pq.toArray(a)
            val max = detections[0]
            nmsRecognitions.add(max)
            pq.clear()
            for (k in 1 until detections.size) {
                val detection = detections[k]
                if (boxIou(
                        max.getLocation(),
                        detection.getLocation()
                    ) < IOU_CLASS_DUPLICATED_THRESHOLD
                ) {
                    pq.add(detection)
                }
            }
        }
        return nmsRecognitions
    }

    protected fun boxIou(a: RectF, b: RectF): Float {
        val intersection = boxIntersection(a, b)
        val union = boxUnion(a, b)
        return if (union <= 0) 1f else intersection / union
    }

    protected fun boxIntersection(a: RectF, b: RectF): Float {
        val maxLeft = if (a.left > b.left) a.left else b.left
        val maxTop = if (a.top > b.top) a.top else b.top
        val minRight = if (a.right < b.right) a.right else b.right
        val minBottom = if (a.bottom < b.bottom) a.bottom else b.bottom
        val w = minRight - maxLeft
        val h = minBottom - maxTop
        return if (w < 0 || h < 0) 0f else w * h
    }

    protected fun boxUnion(a: RectF, b: RectF): Float {
        val i = boxIntersection(a, b)
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i
    }

    /**
     * Добавление прокси-сервера NNapi
     */
    fun addNNApiDelegate() {
        var nnApiDelegate: NnApiDelegate? = null
        // Initialize interpreter with NNAPI delegate for Android Pie or above
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
//            NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
//            nnApiOptions.setAllowFp16(true);
//            nnApiOptions.setUseNnapiCpu(true);
            //ANEURALNETWORKS_PREFER_LOW_POWER：倾向于以最大限度减少电池消耗的方式执行。这种设置适合经常执行的编译。
            //ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER：倾向于尽快返回单个答案，即使这会耗费更多电量。这是默认值。
            //ANEURALNETWORKS_PREFER_SUSTAINED_SPEED：倾向于最大限度地提高连续帧的吞吐量，例如，在处理来自相机的连续帧时。
//            nnApiOptions.setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED);
//            nnApiDelegate = new NnApiDelegate(nnApiOptions);
            nnApiDelegate = NnApiDelegate()
            options.addDelegate(nnApiDelegate)
            Log.i("tfliteSupport", "using nnapi delegate.")
        }
    }

    /**
     * Добавление прокси-серверов GPU
     */
    fun addGPUDelegate() {
        val compatibilityList = CompatibilityList()
        if (compatibilityList.isDelegateSupportedOnThisDevice) {
            val delegateOptions = compatibilityList.bestOptionsForThisDevice
            val gpuDelegate = GpuDelegate(delegateOptions)
            options.addDelegate(gpuDelegate)
            Log.i("tfliteSupport", "using gpu delegate.")
        } else {
            addThread(4)
        }
    }

    /**
     * Количество добавленных threads
     * @param thread
     */
    fun addThread(thread: Int) {
        options.numThreads = thread
    }
}