package com.mobiinfer.demo

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.util.Log

/**
 * Keeps debug-only logging classes out of release APKs. Both the log store and
 * viewer live in `src/debug`, so release builds simply find no implementation.
 */
object DebugLogBridge {
    private const val TAG = "DebugLogBridge"
    private const val STORE_CLASS = "com.mobiinfer.demo.DebugLogStore"
    private const val ACTIVITY_CLASS = "com.mobiinfer.demo.DebugLogActivity"

    private fun storeClass(): Class<*>? =
        try {
            Class.forName(STORE_CLASS)
        } catch (_: ClassNotFoundException) {
            null
        }

    fun isAvailable(): Boolean = storeClass() != null

    fun install(context: Context) {
        try {
            invoke(
                "install",
                arrayOf(Context::class.java),
                arrayOf(context.applicationContext)
            )
        } catch (error: Throwable) {
            Log.w(TAG, "Unable to install debug logging", error)
        }
    }

    fun isEnabled(context: Context): Boolean =
        try {
            invoke(
                "isEnabled",
                arrayOf(Context::class.java),
                arrayOf(context.applicationContext)
            ) as? Boolean ?: false
        } catch (_: Throwable) {
            false
        }

    fun setEnabled(context: Context, enabled: Boolean) {
        try {
            invoke(
                "setEnabled",
                arrayOf(Context::class.java, Boolean::class.javaPrimitiveType!!),
                arrayOf(context.applicationContext, enabled)
            )
        } catch (error: Throwable) {
            Log.w(TAG, "Unable to change debug logging state", error)
        }
    }

    fun open(activity: Activity): Boolean =
        try {
            val clazz = Class.forName(ACTIVITY_CLASS)
            activity.startActivity(Intent(activity, clazz))
            true
        } catch (error: Throwable) {
            Log.w(TAG, "Debug log viewer is not available", error)
            false
        }

    private fun invoke(
        name: String,
        parameterTypes: Array<Class<*>>,
        values: Array<Any?>
    ): Any? {
        val clazz = storeClass() ?: return null
        return clazz.getMethod(name, *parameterTypes).invoke(null, *values)
    }
}
