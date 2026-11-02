package com.mobiinfer.demo

import android.app.Application

class MiniCPMApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        LocaleManager.applyOnAppStart(this)
        DebugLogBridge.install(this)
        // Persistent pipeline trail: logcat is flooded by the DSP driver while
        // an image model runs, and process deaths must not erase the evidence.
        CriticalLog.install(this)
    }
}
