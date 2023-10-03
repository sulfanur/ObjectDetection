package com.example.yolov5tfliteandroid

import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.Surface
import android.view.View
import android.widget.AdapterView
import android.widget.CompoundButton
import android.widget.ImageView
import android.widget.Spinner
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import com.example.yolov5tfliteandroid.analysis.FullImageAnalyse
import com.example.yolov5tfliteandroid.analysis.FullScreenAnalyse
import com.example.yolov5tfliteandroid.detector.Yolov5TFLiteDetector
import com.example.yolov5tfliteandroid.utils.CameraProcess
import com.google.common.util.concurrent.ListenableFuture

class MainActivity : AppCompatActivity() {
    private var IS_FULL_SCREEN = false
    private var cameraPreviewMatch: PreviewView? = null
    private var cameraPreviewWrap: PreviewView? = null
    private var boxLabelCanvas: ImageView? = null
    private var modelSpinner: Spinner? = null
    private var immersive: Switch? = null
    private var inferenceTimeTextView: TextView? = null
    private var frameSizeTextView: TextView? = null
    private var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>? = null
    private var yolov5TFLiteDetector: Yolov5TFLiteDetector? = null
    private val cameraProcess = CameraProcess()
    protected val screenOrientation: Int
        /**
         * 获取屏幕旋转角度,0表示拍照出来的图片是横屏
         *
         */
        protected get() = when (windowManager.defaultDisplay.rotation) {
            Surface.ROTATION_270 -> 270
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_90 -> 90
            else -> 0
        }

    /**
     * 加载模型
     *
     * @param modelName
     */
    private fun initModel(modelName: String) {
        // 加载模型
        try {
            yolov5TFLiteDetector = Yolov5TFLiteDetector()
//            yolov5TFLiteDetector!!.modelFile = modelName
            //            this.yolov5TFLiteDetector.addNNApiDelegate();
            yolov5TFLiteDetector!!.addGPUDelegate()
            yolov5TFLiteDetector!!.initialModel(this)
            Log.i("model", "Success loading model" + yolov5TFLiteDetector!!.initialModel(this))

        } catch (e: Exception) {
            Log.e("image", "load model error: " + e.message + e.toString())
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 打开app的时候隐藏顶部状态栏
//        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_STABLE | View.SYSTEM_UI_FLAG_FULLSCREEN | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN);
        window.decorView.systemUiVisibility =
            View.SYSTEM_UI_FLAG_LAYOUT_STABLE or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
        window.statusBarColor = Color.TRANSPARENT

        // 全屏画面
        cameraPreviewMatch = findViewById(R.id.camera_preview_match)

        // 全图画面
        cameraPreviewWrap = findViewById(R.id.camera_preview_wrap)
        //        cameraPreviewWrap.setScaleType(PreviewView.ScaleType.FILL_START);

        // box/label画面
        boxLabelCanvas = findViewById(R.id.box_label_canvas)

        // 下拉按钮
        modelSpinner = findViewById(R.id.model)

        // 沉浸式体验按钮
        immersive = findViewById(R.id.immersive)

        // 实时更新的一些view
        inferenceTimeTextView = findViewById(R.id.inference_time)
        frameSizeTextView = findViewById(R.id.frame_size)
        cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        // 申请摄像头权限
        if (!cameraProcess.allPermissionsGranted(this)) {
            cameraProcess.requestPermissions(this)
        }

        // 获取手机摄像头拍照旋转参数
        val rotation = windowManager.defaultDisplay.rotation
        Log.i("image", "rotation: $rotation")
        cameraProcess.showCameraSupportSize(this@MainActivity)

        // 初始化加载yolov5s
        initModel("yolov5s")

        // 监听模型切换按钮

        modelSpinner!!.setOnItemSelectedListener(object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(adapterView: AdapterView<*>, view: View, i: Int, l: Long) {
                val model = adapterView.getItemAtPosition(i) as String
                Toast.makeText(this@MainActivity, "loading model: $model", Toast.LENGTH_LONG).show()
                initModel(model)
                if (IS_FULL_SCREEN) {

                    cameraPreviewWrap!!.removeAllViews()
                    val fullScreenAnalyse = FullScreenAnalyse(
                        this@MainActivity,
                        cameraPreviewMatch!!,
                        boxLabelCanvas!!,
                        rotation,
                        inferenceTimeTextView!!,
                        frameSizeTextView!!,
                        yolov5TFLiteDetector!!
                    )
                    cameraProcess.startCamera(
                        this@MainActivity,
                        fullScreenAnalyse,
                        cameraPreviewMatch!!
                    )
                } else {
                    cameraPreviewMatch!!.removeAllViews()
                    val fullImageAnalyse = FullImageAnalyse(
                        this@MainActivity,
                        cameraPreviewWrap!!,
                        boxLabelCanvas!!,
                        rotation,
                        inferenceTimeTextView!!,
                        frameSizeTextView!!,
                        yolov5TFLiteDetector!!
                    )
                    cameraProcess.startCamera(
                        this@MainActivity,
                        fullImageAnalyse,
                        cameraPreviewWrap!!
                    )
                }
            }

            override fun onNothingSelected(adapterView: AdapterView<*>?) {}
        })

        // 监听视图变化按钮
        immersive!!.setOnCheckedChangeListener(CompoundButton.OnCheckedChangeListener { compoundButton, b ->
            IS_FULL_SCREEN = b
            if (b) {
                // 进入全屏模式
                cameraPreviewWrap!!.removeAllViews()
                val fullScreenAnalyse = FullScreenAnalyse(
                    this@MainActivity,
                    cameraPreviewMatch!!,
                    boxLabelCanvas!!,
                    rotation,
                    inferenceTimeTextView!!,
                    frameSizeTextView!!,
                    yolov5TFLiteDetector!!
                )
                cameraProcess.startCamera(this@MainActivity, fullScreenAnalyse, cameraPreviewMatch!!)
            } else {
                // 进入全图模式
                cameraPreviewMatch!!.removeAllViews()
                val fullImageAnalyse = FullImageAnalyse(
                    this@MainActivity,
                    cameraPreviewWrap!!,
                    boxLabelCanvas!!,
                    rotation,
                    inferenceTimeTextView!!,
                    frameSizeTextView!!,
                    yolov5TFLiteDetector!!
                )
                cameraProcess.startCamera(this@MainActivity, fullImageAnalyse, cameraPreviewWrap!!)
            }
        })
    }
}