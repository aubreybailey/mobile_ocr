package io.ente.mobile_ocr

import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession

class TextRegionDetector(modelFiles: DetectionModelFiles) {
    companion object {
        private const val MINIMUM_DETECTION_CONFIDENCE = 0.5f
        private const val MAX_REGIONS = 1000
        private const val DEBUG_TAG = "OnnxOcrDebug"
    }

    private val ortEnv = OrtEnvironment.getEnvironment()
    private val sessionOptions = OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
        // Was CPU-only. NNAPI is a Java-level ORT EP already bundled in the
        // onnxruntime-android AAR -- no NDK/native rebuild needed. Falls
        // back to CPU per-op if a device's NNAPI driver can't run something.
        try {
            addNnapi()
        } catch (e: OrtException) {
            Log.w(DEBUG_TAG, "NNAPI EP unavailable, falling back to CPU", e)
        }
    }
    private val detectionSession = ortEnv.createSession(
        modelFiles.detectionModel.absolutePath,
        sessionOptions
    )

    fun detect(
        bitmap: Bitmap,
        cancellationSignal: OnnxCancellationSignal? = null
    ): List<DetectionCandidate> {
        return TextDetector(detectionSession, ortEnv, cancellationSignal)
            .collectHighConfidenceDetections(
                bitmap = bitmap,
                minimumDetectionConfidence = MINIMUM_DETECTION_CONFIDENCE,
                maxCandidates = MAX_REGIONS
            )
            .candidates
    }

    fun close() {
        detectionSession.close()
        sessionOptions.close()
    }
}
