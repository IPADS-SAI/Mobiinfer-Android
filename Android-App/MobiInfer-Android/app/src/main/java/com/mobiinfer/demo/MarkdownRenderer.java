package com.mobiinfer.demo;

import android.content.Context;
import android.content.res.Configuration;

import io.noties.markwon.Markwon;
import io.noties.markwon.syntax.Prism4jTheme;
import io.noties.markwon.syntax.Prism4jThemeDarkula;
import io.noties.markwon.syntax.Prism4jThemeDefault;
import io.noties.markwon.syntax.SyntaxHighlightPlugin;
import io.noties.prism4j.Prism4j;
import io.noties.prism4j.annotations.PrismBundle;

// Bundle only Python; other code blocks retain ordinary Markdown formatting.
@PrismBundle(include = {"python"}, grammarLocatorClassName = ".PythonGrammarLocator")
public final class MarkdownRenderer {
    private MarkdownRenderer() {}

    public static Markwon create(Context context) {
        boolean dark = (context.getResources().getConfiguration().uiMode
                & Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES;
        Prism4jTheme theme = dark ? Prism4jThemeDarkula.create() : Prism4jThemeDefault.create();
        return Markwon.builder(context)
                .usePlugin(SyntaxHighlightPlugin.create(new Prism4j(new PythonGrammarLocator()), theme))
                .build();
    }
}
