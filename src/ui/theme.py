"""Apparence de l'interface Gradio : titre, CSS et thème sombre."""

TITLE = "ALPR Engine - LEAD Edition"

CSS = """
.gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.glass-card {
    background: rgba(30, 41, 59, 0.8);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 16px;
    padding: 20px;
}

.title-main {
    text-align: center;
    padding: 20px;
}

.subtitle-main {
    text-align: center;
    color: #94a3b8;
}

.access-banner {
    padding: 15px;
    border-radius: 8px;
    font-size: 18px;
    font-weight: bold;
}
"""


# Le CSS peint le fond en bleu nuit : il faut le thème sombre de Gradio pour
# que les textes soient clairs. Ce script l'active au chargement de la page
# (équivalent de l'adresse « ?__theme=dark »).
FORCE_DARK_JS = """
() => {
    const url = new URL(window.location.href);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.replace(url.href);
    }
}
"""
