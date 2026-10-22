package com.mobiinfer.demo

import android.app.Application

class MiniCPMApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        LocaleManager.applyOnAppStart(this)
        DebugLogBridge.install(this)
    }
}
