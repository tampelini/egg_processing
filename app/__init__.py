# app/__init__.py
import os
from flask import Flask

def create_app():
    # caminhos base
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../egg_processing/app
    PROJECT_ROOT = os.path.dirname(BASE_DIR)                # .../egg_processing

    # inicializa app com caminhos corretos
    app = Flask(
        __name__,
        template_folder=os.path.join(BASE_DIR, "templates"),
        static_folder=os.path.join(PROJECT_ROOT, "static")
    )

    # chave secreta (necessária se você quiser usar flash messages ou sessões)
    app.config['SECRET_KEY'] = "um-segredo-bem-seguro"

    # registra rotas
    from .routes import bp as routes_bp
    app.register_blueprint(routes_bp)

    return app
