from flask import Flask
from app.config import config
from app.models import Reader

from elasticsearch import Elasticsearch


def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    host = config[config_name].ELASTIC_URL
    port = config[config_name].ELASTIC_PORT
    index = config[config_name].ELASTIC_INDEX

    es_client = Elasticsearch(hosts=[{"host": host, "port": port}])

    app.reader = Reader(es_client, index)

    from app.main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app