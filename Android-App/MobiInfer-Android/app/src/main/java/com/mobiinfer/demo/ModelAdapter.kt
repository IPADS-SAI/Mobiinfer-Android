package com.mobiinfer.demo

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import com.google.android.material.progressindicator.LinearProgressIndicator

enum class ModelRowState { NOT_DOWNLOADED, DOWNLOADING, PAUSED, READY }

data class ModelRowUi(
    val state: ModelRowState = ModelRowState.NOT_DOWNLOADED,
    val fraction: Float = 0f,
    val message: String? = null,
    /** True when this model is the one currently loaded in the engine. */
    val running: Boolean = false
)

class ModelAdapter(
    private val models: List<ModelInfo>,
    private val selectedModelId: String,
    private val onModelSelected: (ModelInfo) -> Unit,
    private val onDownload: (ModelInfo) -> Unit,
    private val onPause: (ModelInfo) -> Unit,
    private val onResume: (ModelInfo) -> Unit,
    private val onDelete: (ModelInfo) -> Unit
) : RecyclerView.Adapter<ModelAdapter.ViewHolder>() {

    private var currentSelectedId = selectedModelId
    private val states = mutableMapOf<String, ModelRowUi>()
    private var progressHolder: ViewHolder? = null
    private var progressModelId: String? = null

    class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val tvName: TextView = itemView.findViewById(R.id.tv_model_name)
        val tvDesc: TextView = itemView.findViewById(R.id.tv_model_desc)
        val tvState: TextView = itemView.findViewById(R.id.tv_model_state)
        val ivSelected: ImageView = itemView.findViewById(R.id.iv_selected)
        val tvRunning: TextView = itemView.findViewById(R.id.tv_running)
        val progress: LinearProgressIndicator = itemView.findViewById(R.id.progress_model)
        val btnDownload: MaterialButton = itemView.findViewById(R.id.btn_model_download)
        val btnPause: MaterialButton = itemView.findViewById(R.id.btn_model_pause)
        val btnResume: MaterialButton = itemView.findViewById(R.id.btn_model_resume)
        val btnDelete: MaterialButton = itemView.findViewById(R.id.btn_model_delete)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_model_card, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val model = models[position]
        val ui = states[model.id] ?: ModelRowUi()
        val ctx = holder.itemView.context

        holder.tvName.text = model.displayName
        holder.tvDesc.text = model.getDescription(ctx)

        val isSelected = model.id == currentSelectedId
        holder.ivSelected.setImageResource(
            if (isSelected) R.drawable.ic_radio_on else R.drawable.ic_radio_off
        )
        holder.tvRunning.visibility = if (ui.running) View.VISIBLE else View.GONE

        val showProgress = ui.state == ModelRowState.DOWNLOADING || ui.state == ModelRowState.PAUSED
        holder.progress.visibility = if (showProgress) View.VISIBLE else View.GONE
        if (showProgress) {
            holder.progress.isIndeterminate = false
            holder.progress.setProgressCompat((ui.fraction * 100f).toInt().coerceIn(0, 100), false)
        }

        when (ui.state) {
            ModelRowState.NOT_DOWNLOADED -> {
                holder.tvState.visibility = View.GONE
                holder.btnDownload.visibility = View.VISIBLE
                holder.btnPause.visibility = View.GONE
                holder.btnResume.visibility = View.GONE
                holder.btnDelete.visibility = View.GONE
            }
            ModelRowState.DOWNLOADING -> {
                holder.tvState.visibility = View.VISIBLE
                holder.tvState.text = ui.message ?: ctx.getString(R.string.status_downloading)
                holder.btnDownload.visibility = View.GONE
                holder.btnPause.visibility = View.VISIBLE
                holder.btnResume.visibility = View.GONE
                holder.btnDelete.visibility = View.GONE
            }
            ModelRowState.PAUSED -> {
                holder.tvState.visibility = View.VISIBLE
                holder.tvState.text = ctx.getString(
                    R.string.model_paused_progress,
                    (ui.fraction * 100f).toInt().coerceIn(0, 100)
                )
                holder.btnDownload.visibility = View.GONE
                holder.btnPause.visibility = View.GONE
                holder.btnResume.visibility = View.VISIBLE
                holder.btnDelete.visibility = View.VISIBLE
            }
            ModelRowState.READY -> {
                holder.tvState.visibility = View.VISIBLE
                holder.tvState.text = ctx.getString(R.string.model_ready_state)
                holder.btnDownload.visibility = View.GONE
                holder.btnPause.visibility = View.GONE
                holder.btnResume.visibility = View.GONE
                holder.btnDelete.visibility = View.VISIBLE
            }
        }

        if (ui.state == ModelRowState.DOWNLOADING && ui.message == null) {
            holder.tvState.text = ctx.getString(R.string.status_downloading)
        }

        holder.itemView.setOnClickListener {
            if (model.id != currentSelectedId) {
                val oldIndex = models.indexOfFirst { it.id == currentSelectedId }
                currentSelectedId = model.id
                if (oldIndex >= 0) notifyItemChanged(oldIndex)
                notifyItemChanged(position)
                onModelSelected(model)
            }
        }

        holder.btnDownload.setOnClickListener { onDownload(model) }
        holder.btnPause.setOnClickListener { onPause(model) }
        holder.btnResume.setOnClickListener { onResume(model) }
        holder.btnDelete.setOnClickListener { onDelete(model) }

        if (model.id == progressModelId) {
            progressHolder = holder
        }
    }

    override fun onViewRecycled(holder: ViewHolder) {
        if (holder == progressHolder) progressHolder = null
    }

    override fun getItemCount(): Int = models.size

    fun updateSelection(modelId: String) {
        val oldIndex = models.indexOfFirst { it.id == currentSelectedId }
        currentSelectedId = modelId
        if (oldIndex >= 0) notifyItemChanged(oldIndex)
        val newIndex = models.indexOfFirst { it.id == modelId }
        if (newIndex >= 0) notifyItemChanged(newIndex)
    }

    fun setStates(newStates: Map<String, ModelRowUi>) {
        states.clear()
        states.putAll(newStates)
        notifyDataSetChanged()
    }

    fun updateRow(modelId: String, ui: ModelRowUi) {
        states[modelId] = ui
        val index = models.indexOfFirst { it.id == modelId }
        if (index >= 0) notifyItemChanged(index)
    }

    fun setProgressTarget(modelId: String?) {
        progressModelId = modelId
        progressHolder = null
    }

    /** In-place progress update to avoid rebinding/flicker during download. */
    fun updateProgress(modelId: String, fraction: Float, message: String?) {
        states[modelId] = (states[modelId] ?: ModelRowUi(state = ModelRowState.DOWNLOADING))
            .copy(state = ModelRowState.DOWNLOADING, fraction = fraction, message = message)
        if (modelId == progressModelId) {
            progressHolder?.let { holder ->
                holder.progress.visibility = View.VISIBLE
                holder.progress.setProgressCompat((fraction * 100f).toInt().coerceIn(0, 100), false)
                holder.tvState.visibility = View.VISIBLE
                holder.tvState.text = message ?: holder.itemView.context.getString(R.string.status_downloading)
            }
        }
    }
}
