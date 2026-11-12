import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from .config import settings

log = logging.getLogger(__name__)
class Base(DeclarativeBase): pass
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
def init_db():
    from . import models  # register mapped tables
    try:
        Base.metadata.create_all(engine)
    except Exception:
        log.exception('Database initialization failed')
        raise
def get_db():
    db=SessionLocal()
    try: yield db
    finally: db.close()
def healthcheck():
    with engine.connect() as conn: conn.execute(text('SELECT 1'))
