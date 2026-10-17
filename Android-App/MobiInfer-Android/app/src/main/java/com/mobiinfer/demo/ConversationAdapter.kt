package com.mobiinfer.demo

import android.graphics.Typeface
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Rows of the conversation drawer. Tap opens, long-press renames, trash deletes. */
class ConversationAdapter(
    private val onOpen: (ConversationMeta) -> Unit,
    private val onDelete: (ConversationMeta) -> Unit,
    private val onRename: (ConversationMeta) -> Unit
) : RecyclerView.Adapter<ConversationAdapter.VH>() {

    private val items = mutableListOf<ConversationMeta>()
    private var selectedId: String? = null
    private val fmt = SimpleDateFormat("MM-dd HH:mm", Locale.getDefault())

    fun submit(list: List<ConversationMeta>, selected: String?) {
        items.clear()
        items.addAll(list)
        selectedId = selected
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val v = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_conversation, parent, false)
        return VH(v)
    }

    override fun getItemCount() = items.size

    override fun onBindViewHolder(holder: VH, position: Int) = holder.bind(items[position])

    inner class VH(v: View) : RecyclerView.ViewHolder(v) {
        private val title: TextView = v.findViewById(R.id.tv_conv_title)
        private val time: TextView = v.findViewById(R.id.tv_conv_time)
        private val delete: ImageButton = v.findViewById(R.id.btn_conv_delete)

        fun bind(meta: ConversationMeta) {
            title.text = meta.title
            time.text = fmt.format(Date(meta.updatedAt))
            val selected = meta.id == selectedId
            title.setTypeface(null, if (selected) Typeface.BOLD else Typeface.NORMAL)
            itemView.setOnClickListener { onOpen(meta) }
            itemView.setOnLongClickListener { onRename(meta); true }
            delete.setOnClickListener { onDelete(meta) }
        }
    }
}
