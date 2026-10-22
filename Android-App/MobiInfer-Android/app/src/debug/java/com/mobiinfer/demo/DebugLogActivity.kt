package com.mobiinfer.demo

import android.content.ClipData
import android.content.ClipboardManager
import android.os.Bundle
import android.view.View
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import com.google.android.material.button.MaterialButton

class DebugLogActivity : AppCompatActivity() {
    private lateinit var scrollLog: ScrollView
    private lateinit var tvLog: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_debug_log)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        val rootContent = findViewById<View>(android.R.id.content)
        ViewCompat.setOnApplyWindowInsetsListener(rootContent) { view, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.updatePadding(
                left = systemBars.left,
                top = systemBars.top,
                right = systemBars.right,
                bottom = systemBars.bottom
            )
            insets
        }

        findViewById<Toolbar>(R.id.toolbar).setNavigationOnClickListener { finish() }
        scrollLog = findViewById(R.id.scroll_debug_log)
        tvLog = findViewById(R.id.tv_debug_log)

        findViewById<MaterialButton>(R.id.btn_copy_debug_log).setOnClickListener {
            copyLog()
        }
        findViewById<MaterialButton>(R.id.btn_clear_debug_log).setOnClickListener {
            confirmClear()
        }
        refreshLog()
    }

    override fun onResume() {
        super.onResume()
        refreshLog()
    }

    private fun refreshLog() {
        val text = DebugLogStore.readLog(this)
        tvLog.text = text.ifBlank { getString(R.string.debug_log_empty) }
        scrollLog.post { scrollLog.fullScroll(View.FOCUS_DOWN) }
    }

    private fun copyLog() {
        val text = tvLog.text.toString()
        if (text.isBlank()) {
            Toast.makeText(this, R.string.debug_log_empty, Toast.LENGTH_SHORT).show()
            return
        }
        val clipboard = getSystemService(ClipboardManager::class.java)
        clipboard.setPrimaryClip(ClipData.newPlainText("MobiInfer debug log", text))
        Toast.makeText(this, R.string.debug_log_copied, Toast.LENGTH_SHORT).show()
    }

    private fun confirmClear() {
        AlertDialog.Builder(this)
            .setTitle(R.string.debug_log_clear_title)
            .setMessage(R.string.debug_log_clear_message)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.debug_log_clear_action) { _, _ ->
                DebugLogStore.clearLog(this)
                refreshLog()
            }
            .show()
    }
}
