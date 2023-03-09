from pathlib import Path
from shadow_scholar.cli import cli, safe_import, Argument


with safe_import():
    import gradio as gr


S2R_URL = "https://www.semanticscholar.org/research/research-team"
SWS_URL = "https://github.com/allenai/shadow-scholar"
CSS_URI = Path(__file__).parent / "res" / "style.css"
IMG_URI = Path(__file__).parent / "res" / "logo.png"


@cli(
    'app.hello_world',
    arguments=[
        Argument('--host', default='0.0.0.0', help='Host to bind to'),
        Argument('--port', default=7860, help='Port to bind to')
    ],
    requirements=['gradio'],
)
def run_hello_world(host: str, port: int):
    with open(CSS_URI, 'r') as f:
        css = f.read()

    with gr.Blocks(title="Shadow Scholar", css=css) as app:
        gr.Markdown("# 🕶️ 🎓 Shadow Scholar 🎓 🕶️", elem_id="center")
        gr.Image(
            str(IMG_URI),
            elem_id="logo",
            shape=(400, 400),
            show_label=False,
            interactive=False,
        )
        gr.Markdown(
            "Shadow Scholar is a collection of tools and applications "
            f"from the [S2 Research Team]({S2R_URL}).\n\nTo learn more about "
            f"this project, please visit our [GitHub repository]({SWS_URL}).",
            elem_id="center"
        )

    try:
        app.launch(server_name=host, server_port=port, show_api=False)
    except Exception as e:
        gr.close_all()
        raise e
