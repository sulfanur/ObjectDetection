package com.example.yolov5tfliteandroid.utils

import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.util.Log
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import java.util.concurrent.ExecutionException

class CameraProcess {
    private var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>? = null
    private val REQUEST_CODE_PERMISSIONS = 1001
    private val REQUIRED_PERMISSIONS = arrayOf(
        "android.permission.CAMERA",
        "android.permission.WRITE_EXTERNAL_STORAGE"
    )

    /**
     * 判断摄像头权限
     * @param context
     * @return
     */
    fun allPermissionsGranted(context: Context?): Boolean {
        for (permission in REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(
                    context!!,
                    permission
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                return false
            }
        }
        return true
    }

    /**
     * 申请摄像头权限
     * @param activity
     */
    fun requestPermissions(activity: Activity?) {
        ActivityCompat.requestPermissions(
            activity!!,
            REQUIRED_PERMISSIONS,
            REQUEST_CODE_PERMISSIONS
        )
    }

    /**
     * 打开摄像头，提供对应的preview, 并且注册analyse事件, analyse就是要对摄像头每一帧进行分析的操作
     */
    fun startCamera(
        context: Context?,
        analyzer: ImageAnalysis.Analyzer?,
        previewView: PreviewView
    ) {
        cameraProviderFuture = ProcessCameraProvider.getInstance(context!!)
        cameraProviderFuture!!.addListener({
            try {
                val cameraProvider = cameraProviderFuture!!.get()
                val imageAnalysis =
                    ImageAnalysis.Builder() //                            .setTargetResolution(new Size(1080, 1920))
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3) //                            .setTargetAspectRatioCustom(new Rational(16,9))
                        //                            .setTargetRotation(Surface.ROTATION_90)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()
                imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(context), analyzer!!)
                val previewBuilder =
                    Preview.Builder() //                            .setTargetResolution(new Size(1080,1440))
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3) //                            .setTargetRotation(Surface.ROTATION_90)
                        .build()
                //                    Log.i("builder", previewView.getHeight()+"/"+previewView.getWidth());
                val cameraSelector = CameraSelector.Builder()
                    .requireLensFacing(CameraSelector.LENS_FACING_BACK).build()
                previewBuilder.setSurfaceProvider(previewView.createSurfaceProvider())
                // 加多这一步是为了切换不同视图的时候能释放上一视图所有绑定事件
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    (context as LifecycleOwner?)!!,
                    cameraSelector,
                    imageAnalysis,
                    previewBuilder
                )
            } catch (e: ExecutionException) {
                e.printStackTrace()
            } catch (e: InterruptedException) {
                e.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(context))
    }

    /**
     * 打印输出摄像头支持的宽和高
     * @param activity
     */
    fun showCameraSupportSize(activity: Activity) {
        val manager = activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            for (id in manager.cameraIdList) {
                val cc = manager.getCameraCharacteristics(id!!)
                if (cc.get(CameraCharacteristics.LENS_FACING) == 1) {
                    val previewSizes = cc.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                        .getOutputSizes(SurfaceTexture::class.java)
                    for (s in previewSizes) {
                        Log.i("camera", s.height.toString() + "/" + s.width)
                    }
                    break
                }
            }
        } catch (e: Exception) {
            Log.e("image", "can not open camera", e)
        }
    }
}